import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------
# Configuración base
# ---------------------------
SITE_URL = "https://dataky.net"

# Probamos sitemaps comunes (plugins SEO suelen usar sitemap_index.xml)
SITEMAPS = [
    f"{SITE_URL}/sitemap_index.xml",
    f"{SITE_URL}/wp-sitemap.xml",
    f"{SITE_URL}/sitemap.xml",
]

# ---------------------------
# HTTP: descarga contenido de una URL
# ---------------------------
def fetch(url: str) -> str:
    """
    Descarga texto (HTML o XML) desde una URL.
    - timeout evita que se quede colgado.
    - User-Agent ayuda a evitar bloqueos básicos.
    """
    r = requests.get(url, timeout=25, headers={"User-Agent": "DatakyBot/1.0"})
    r.raise_for_status()  # si es 404/500 lanza excepción
    return r.text

# ---------------------------
# Helpers de validación de dominio
# ---------------------------
def same_domain(url: str, base: str) -> bool:
    """
    Valida que la URL pertenezca al mismo dominio que SITE_URL.
    Esto evita que el sitemap nos mande a dominios externos.
    """
    return urlparse(url).netloc == urlparse(base).netloc

# ---------------------------
# Parseo XML: extraer <loc>...</loc>
# ---------------------------
def extract_locs(xml_text: str) -> list[str]:
    """
    Un sitemap (o sitemap index) tiene URLs dentro de etiquetas <loc>.
    BeautifulSoup con parser 'xml' facilita extraerlas.
    """
    soup = BeautifulSoup(xml_text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]

# ---------------------------
# Elegir el sitemap que realmente existe
# ---------------------------
def pick_working_sitemap() -> str:
    """
    Recorre posibles rutas de sitemap y devuelve la primera que responda OK.
    """
    for sm in SITEMAPS:
        try:
            _ = fetch(sm)
            return sm
        except Exception:
            pass
    raise RuntimeError(
        "No encontré un sitemap accesible en /sitemap_index.xml, /wp-sitemap.xml o /sitemap.xml"
    )


# ---------------------------
# Expandir sitemap index (lista de sitemaps)
# ---------------------------
def expand_if_index(locs: list[str]) -> list[str]:
    """
    Si el sitemap es un 'index', los <loc> apuntan a otros XML.
    Ej:
      <loc>https://dominio.com/post-sitemap.xml</loc>

    Entonces descargamos esos sub-sitemaps y extraemos sus URLs finales.
    """
    is_index = any(u.endswith(".xml") for u in locs) and len(locs) < 10000

    if is_index:
        urls = []
        for sub in locs[:40]:  # límite de seguridad para no tardar demasiado
            try:
                sub_xml = fetch(sub)
                urls.extend(extract_locs(sub_xml))
            except Exception:
                # si un sub-sitemap falla, no tumba todo el proceso
                pass
        return sorted(set(urls))

    # Si NO es index, ya es un urlset "final"
    return sorted(set(locs))

# ---------------------------
# Filtro de URLs útiles (customizable)
# ---------------------------
def useful(url: str) -> bool:
    """
    Define qué URLs son útiles para el chatbot.

    En general:
    - Excluir páginas de taxonomía / archive que meten ruido
    - Excluir endpoints técnicos (wp-json, feeds)
    - Excluir archivos (imágenes, pdfs, zip)
    - Priorizar páginas con intención comercial (productos/servicios/contacto)
    """
    u = url.lower()

    # 1) excluir basura típica
    bad_contains = [
        "/tag/", "/category/", "/author/", "/feed/",
        "/wp-json/", "/xmlrpc.php",
        "?replytocom=",
    ]
    if any(b in u for b in bad_contains):
        return False

    # 2) excluir archivos (para el MVP; luego puedes incluir pdf si quieres)
    if re.search(r"\.(jpg|jpeg|png|webp|gif|pdf|zip|rar)$", u):
        return False

    # 3) excluir páginas técnicas o poco útiles para venta
    bad_exact_or_contains = ["/cart/", "/checkout/", "/my-account/"]
    if any(b in u for b in bad_exact_or_contains):
        return False

    # 4) preferir páginas “comerciales”
    good_contains = [
        "/product/",   # WooCommerce en inglés (default)
        "/producto/",  # si tu slug está en español
        "/serv",       # /servicios/, /service/
        "/contact",    # /contacto/
        "/curso",      # /curso/, /cursos/
        "/pack",       # /packs/
        "/oferta",     # landing / oferta
    ]

    # Condición final:
    # - La URL debe ser del sitio y contener alguno de los patrones buenos
    return u.startswith(SITE_URL.lower()) and any(g in u for g in good_contains)


# ---------------------------
# Programa principal
# ---------------------------
def main():
    sm = pick_working_sitemap()
    print(f"✅ Sitemap usado: {sm}")

    xml = fetch(sm)
    locs = extract_locs(xml)

    # Si es sitemap index, expandimos; si no, ya tenemos URLs finales
    urls = expand_if_index(locs)

    # Filtra a mismo dominio
    urls = [u for u in urls if same_domain(u, SITE_URL)]
    print(f"🔎 URLs totales encontradas: {len(urls)}")

    # Filtra URLs útiles para el chatbot
    useful_urls = [u for u in urls if useful(u)]
    print(f"✅ URLs útiles tras filtro: {len(useful_urls)}\n")

    # Vista previa
    print("📌 Ejemplo (primeras 20):")
    for u in useful_urls[:20]:
        print(" -", u)

    # Guarda el resultado para el siguiente módulo (extracción de texto)
    import os
    os.makedirs("data", exist_ok=True)

    with open("data/urls_filtradas.txt", "w", encoding="utf-8") as f:
        for u in useful_urls:
            f.write(u + "\n")

    print("\n💾 Guardado en: data/urls_filtradas.txt")


if __name__ == "__main__":
    main()