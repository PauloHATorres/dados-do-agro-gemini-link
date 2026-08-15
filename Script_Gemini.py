"""Classifica os registros de um CSV com a API do Gemini.

Pode ser executado diretamente ou importado no final do script Selenium.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field


PASTA_DO_PROJETO = Path(__file__).resolve().parent
NOME_ARQUIVO_CSV = "arquivo.csv"
MODELO_PADRAO = "gemini-3.6-flash"
TAMANHO_LOTE_PADRAO = 100

COLUNA_TEMA = "Classificação temática"
COLUNA_FONTE = "Classificação da fonte"

CATEGORIAS_FONTE = (
    "Institucional",
    "Particular",
    "Pesquisa científica",
    "Organização não governamental",
    "Não identificado",
)

PROMPT_PADRAO = """
Você classifica conjuntos de dados relacionados ao setor agropecuário.

Para cada registro recebido, produza exatamente uma classificação temática e
uma classificação da fonte.

Classificação temática:
- Identifique o assunto principal do conjunto de dados.
- Use uma categoria curta, objetiva e em português.
- Sempre que possível, reutilize categorias como: Produção, Plantel ou
  rebanho, Banco de imagens, Preços, Comércio, Saúde animal, Genética,
  Alimentação, Meio ambiente, Clima e Outros.
- Considere o tema geral da busca informado pelo usuário.

Classificação da fonte:
- Use somente uma destas categorias: Institucional, Particular,
  Pesquisa científica, Organização não governamental ou Não identificado.
- Considere principalmente título, link, fornecedor, licença e descrição.

Não altere nem resuma os dados originais. Devolva uma classificação para
cada id_linha recebido e preserve exatamente esse identificador.
""".strip()


class Classificacao(BaseModel):
    """Formato obrigatório de cada item devolvido pelo Gemini."""

    id_linha: int = Field(description="Identificador numérico recebido na entrada")
    classificacao_tematica: str
    classificacao_fonte: str


def _ler_csv(caminho: Path) -> tuple[list[dict[str, str]], list[str], str]:
    """Lê o CSV, detectando codificação e separador."""
    ultimo_erro: UnicodeDecodeError | None = None

    for codificacao in ("utf-8-sig", "latin-1"):
        try:
            conteudo = caminho.read_text(encoding=codificacao)
            break
        except UnicodeDecodeError as erro:
            ultimo_erro = erro
    else:
        raise ValueError(f"Não foi possível ler a codificação do CSV: {ultimo_erro}")

    try:
        separador = csv.Sniffer().sniff(conteudo[:8192], delimiters=";,\t|").delimiter
    except csv.Error:
        separador = ";"

    leitor = csv.DictReader(conteudo.splitlines(), delimiter=separador)
    if not leitor.fieldnames:
        raise ValueError("O CSV não possui cabeçalho.")

    cabecalhos = [campo.strip() for campo in leitor.fieldnames]
    linhas: list[dict[str, str]] = []
    for linha in leitor:
        linhas.append(
            {
                cabecalho: (valor or "").strip()
                for cabecalho, valor in zip(cabecalhos, linha.values())
            }
        )

    if not linhas:
        raise ValueError("O CSV não possui linhas de dados.")

    return linhas, cabecalhos, separador


def _classificar_lote(
    cliente: genai.Client,
    modelo: str,
    tema_busca: str,
    registros: list[dict[str, object]],
    tentativas: int = 3,
) -> list[Classificacao]:
    entrada = json.dumps(registros, ensure_ascii=False)
    instrucao = (
        f"{PROMPT_PADRAO}\n\n"
        f"Tema geral/string usada na busca: {tema_busca or 'não informado'}\n\n"
        f"Registros para classificar:\n{entrada}"
    )

    ultimo_erro: Exception | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = cliente.models.generate_content(
                model=modelo,
                contents=instrucao,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[Classificacao],
                    temperature=0.1,
                ),
            )
            if not resposta.text:
                raise RuntimeError("O Gemini devolveu uma resposta sem texto.")

            dados = json.loads(resposta.text)
            classificacoes = [Classificacao.model_validate(item) for item in dados]

            ids_esperados = {int(item["id_linha"]) for item in registros}
            ids_recebidos = {item.id_linha for item in classificacoes}
            if ids_recebidos != ids_esperados or len(classificacoes) != len(registros):
                raise ValueError(
                    "A resposta não contém exatamente uma classificação por linha."
                )

            for item in classificacoes:
                if item.classificacao_fonte not in CATEGORIAS_FONTE:
                    raise ValueError(
                        f"Categoria de fonte inesperada: {item.classificacao_fonte}"
                    )
            return classificacoes
        except ClientError as erro:
            # Erros 4xx normalmente indicam chave, modelo ou requisição inválida;
            # repetir a mesma chamada não resolveria o problema.
            if 400 <= erro.code < 500:
                if erro.code == 404:
                    raise RuntimeError(
                        f"O modelo '{modelo}' não está disponível para esta conta. "
                        "Defina outro modelo na variável GEMINI_MODEL."
                    ) from erro
                if erro.code in (401, 403):
                    raise RuntimeError(
                        "A chave GEMINI_API_KEY foi recusada. Confira se ela está "
                        "ativa e se possui acesso à API do Gemini."
                    ) from erro
                raise RuntimeError(f"A API do Gemini recusou a requisição: {erro}") from erro
            ultimo_erro = erro
            if tentativa < tentativas:
                time.sleep(2 ** (tentativa - 1))
        except Exception as erro:
            ultimo_erro = erro
            if tentativa < tentativas:
                time.sleep(2 ** (tentativa - 1))

    raise RuntimeError(
        f"Falha ao classificar um lote após {tentativas} tentativas: {ultimo_erro}"
    ) from ultimo_erro


def _escrever_csv_atomico(
    destino: Path,
    linhas: list[dict[str, str]],
    cabecalhos: list[str],
    separador: str,
) -> None:
    """Grava um CSV sem deixar um arquivo incompleto em caso de interrupção."""
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporario.open("w", encoding="utf-8-sig", newline="") as arquivo_saida:
            escritor = csv.DictWriter(
                arquivo_saida,
                fieldnames=cabecalhos,
                delimiter=separador,
                extrasaction="ignore",
            )
            escritor.writeheader()
            escritor.writerows(linhas)
        temporario.replace(destino)
    finally:
        if temporario.exists():
            temporario.unlink()


def classificar_csv(
    caminho_csv: str | Path,
    *,
    tema_busca: str = "",
    caminho_saida: str | Path | None = None,
    modelo: str | None = None,
    tamanho_lote: int = TAMANHO_LOTE_PADRAO,
) -> Path:
    """Classifica o CSV e cria uma cópia com duas novas colunas."""
    arquivo = Path(caminho_csv).expanduser().resolve()
    if not arquivo.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {arquivo}")
    if arquivo.suffix.lower() != ".csv":
        raise ValueError(f"O arquivo precisa ter extensão .csv: {arquivo}")
    if tamanho_lote < 1:
        raise ValueError("O tamanho do lote precisa ser maior que zero.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")

    linhas, cabecalhos, separador = _ler_csv(arquivo)
    destino = (
        Path(caminho_saida).expanduser().resolve()
        if caminho_saida
        else arquivo.with_name(f"{arquivo.stem}_classificado.csv")
    )
    nome_modelo = modelo or os.getenv("GEMINI_MODEL", MODELO_PADRAO)
    assinatura = hashlib.sha256(
        arquivo.read_bytes()
        + tema_busca.encode("utf-8")
        + nome_modelo.encode("utf-8")
        + str(tamanho_lote).encode("ascii")
    ).hexdigest()[:10]
    pasta_partes = arquivo.parent / f"{arquivo.stem}_partes_{assinatura}"
    pasta_partes.mkdir(parents=True, exist_ok=True)
    cliente = genai.Client(api_key=api_key)

    novos_cabecalhos = [
        campo for campo in cabecalhos if campo not in (COLUNA_TEMA, COLUNA_FONTE)
    ] + [COLUNA_TEMA, COLUNA_FONTE]

    try:
        total = len(linhas)
        for inicio in range(0, total, tamanho_lote):
            fim = min(inicio + tamanho_lote, total)
            numero_parte = inicio // tamanho_lote + 1
            caminho_parte = pasta_partes / f"{arquivo.stem}_parte_{numero_parte:03d}.csv"

            if caminho_parte.is_file():
                linhas_parte, _, _ = _ler_csv(caminho_parte)
                quantidade_esperada = fim - inicio
                parte_valida = (
                    len(linhas_parte) == quantidade_esperada
                    and all(
                        linha.get(COLUNA_TEMA) and linha.get(COLUNA_FONTE)
                        for linha in linhas_parte
                    )
                )
                if parte_valida:
                    print(
                        f"Reutilizando parte {numero_parte}: linhas "
                        f"{inicio + 1} a {fim} de {total}."
                    )
                    for deslocamento, linha_parte in enumerate(linhas_parte):
                        linhas[inicio + deslocamento][COLUNA_TEMA] = linha_parte[
                            COLUNA_TEMA
                        ]
                        linhas[inicio + deslocamento][COLUNA_FONTE] = linha_parte[
                            COLUNA_FONTE
                        ]
                    continue

            registros = [
                {"id_linha": indice + 1, **linhas[indice]}
                for indice in range(inicio, fim)
            ]
            print(f"Classificando linhas {inicio + 1} a {fim} de {total}...")
            classificacoes = _classificar_lote(
                cliente, nome_modelo, tema_busca, registros
            )

            for item in classificacoes:
                linha = linhas[item.id_linha - 1]
                linha[COLUNA_TEMA] = item.classificacao_tematica.strip()
                linha[COLUNA_FONTE] = item.classificacao_fonte.strip()

            _escrever_csv_atomico(
                caminho_parte,
                linhas[inicio:fim],
                novos_cabecalhos,
                separador,
            )
            print(f"Parte salva: {caminho_parte.name}")
    finally:
        cliente.close()

    _escrever_csv_atomico(destino, linhas, novos_cabecalhos, separador)

    print(f"Arquivo classificado criado em: {destino}")
    print(f"Partes preservadas para retomada em: {pasta_partes}")
    return destino


def executar_apos_selenium(caminho_csv: str | Path, tema_busca: str = "") -> Path:
    """Ponto de integração para chamar no final do Selenium."""
    return classificar_csv(caminho_csv, tema_busca=tema_busca)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica os registros de um CSV usando o Gemini."
    )
    parser.add_argument(
        "arquivo",
        nargs="?",
        default=PASTA_DO_PROJETO / NOME_ARQUIVO_CSV,
        type=Path,
        help="Caminho do CSV gerado pelo web scraping.",
    )
    parser.add_argument(
        "--tema",
        default="",
        help='Tema/string da busca, por exemplo: "leite".',
    )
    parser.add_argument("--saida", type=Path, help="Caminho opcional do CSV final.")
    parser.add_argument(
        "--lote",
        type=int,
        default=TAMANHO_LOTE_PADRAO,
        help="Quantidade de registros enviada em cada chamada (padrão: 100).",
    )
    args = parser.parse_args()

    classificar_csv(
        args.arquivo,
        tema_busca=args.tema,
        caminho_saida=args.saida,
        tamanho_lote=args.lote,
    )


if __name__ == "__main__":
    main()
