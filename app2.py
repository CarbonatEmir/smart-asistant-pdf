import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import base64
import os

st.set_page_config(
    page_title="Lasersan AI", 
    page_icon="🤖", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# geçiş   #334155

if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False
if "current_product" not in st.session_state:
    st.session_state.current_product = None

urun_secimi = st.query_params.get("product")

#pdf kapat

if urun_secimi != st.session_state.current_product:
    st.session_state.current_product = urun_secimi
    st.session_state.show_pdf = False

custom_css = """
<style>
    /* Genel Arka Plan ve Kararlı Animasyon */
    .stApp {
        background-color: #0f172a !important;
    }
    
    @keyframes fadeInSlideUp {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="column"] {
        animation: fadeInSlideUp 0.4s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* Görsel Boyutlandırma (Çok büyümesini engeller) */
    [data-testid="stImage"] img {
        max-height: 380px !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        filter: drop-shadow(0 15px 25px rgba(0,0,0,0.4));
    }
    
    /* Sol menüdeki logonun boyutunu serbest bırak */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        max-height: none !important;
        filter: none !important;
    }

    /* Chatbot Tasarımı */
    .stChatInputContainer > div {
        border-radius: 24px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        padding-left: 10px;
        background-color: white;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #003B70 !important;
        color: white !important;
    }
    
    /* Başlık Hizalamaları */
    .prod-title-large {
        text-align: center; color: #C8102E; font-size: 3.5rem; font-weight: 900; margin-bottom: 20px; font-family: "Segoe UI", sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-top: 0;
    }
    .prod-title-small {
        text-align: center; color: #C8102E; font-size: 2.2rem; font-weight: 900; margin-bottom: 20px; font-family: "Segoe UI", sans-serif; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); margin-top: 0;
    }
    .fancy-pdf-header {
        text-align: center; font-size: 1.6rem; color: #e2e8f0; margin-top: 0; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px dashed #334155; letter-spacing: 2px; font-family: 'Segoe UI', sans-serif;
    }
    .fancy-pdf-header span {
        color: #C8102E; font-weight: 900;
    }

    /* Sol Menü Tasarımı (Mat Turkuaz) */
    [data-testid="stSidebar"] {
        background-color: #44A2BD!important;   
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #BFEFF8 !important;
        margin-bottom: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary {
        transition: all 0.3s ease !important;
        padding: 8px !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover {
        background-color: #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary p {
        color: #e2e8f0 !important; font-weight: 700 !important; font-size: 0.95rem !important;
    }

    /* Link Tasarımları */
    .sidebar-link-btn {
        display: block; padding: 4px 8px; background-color: #107F93; color: #f8fafc !important; text-align: center; border-radius: 6px; text-decoration: none; font-weight: 700; margin-bottom: 6px; transition: all 0.3s ease; font-size: 0.85rem; border: 1px solid transparent;
    }
    .sidebar-link-btn:hover {
        background-color: #003B70; color: white !important; border: 1px solid #003B70; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* ------ BUTON TASARIMLARI ------ */
    /* Mavi Buton (Broşürü Görüntüle) */
    button[kind="secondary"] {
        background-color: #107F93 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    button[kind="secondary"]:hover {
        background-color: #003B70 !important;
        transform: translateY(-2px);
    }
    
    /* Kırmızı Buton (Broşürü Kapat) */
    button[kind="primary"] {
        background-color: #C8102E !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    button[kind="primary"]:hover {
        background-color: #9e0c24 !important;
        transform: translateY(-2px);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    logo_file = "logo3.png" if os.path.exists("logo3.png") else "logo.png"
    if os.path.exists(logo_file):
        with open(logo_file, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
        st.markdown(
            f'''
            <div style="text-align: center;">
                <a href="/" target="_self" title="Ana Sayfaya Dön">
                    <img src="data:image/png;base64,{b64_logo}" style="width: 80%; cursor: pointer;">
                </a>
            </div>
            ''',
            unsafe_allow_html=True
        )
            
    st.markdown(
        "<div style='text-align: center; color: #e2e8f0; font-family: \"Segoe UI\", sans-serif; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px; margin-bottom: 30px;'>"
        "Lasersan Advanced Technology"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<h3 style='text-align: center; color: #f8fafc; font-weight: 800; font-size: 1.2rem; margin-bottom: 15px;'>"
        "Ürünler"
        "</h3>", 
        unsafe_allow_html=True
    )
    
    kategoriler_ve_urunler = {
        "Gece Görüş Sistemleri": ["AURA", "FOCUS", "ODAK", "TUNAY"],
        "Renkli Gece Görüş Sistemleri": ["NOXIS"],
        "Termal El Dürbünü": ["TALOS"],
        "Silah Üstü Nişangahlar": ["KOZGU"],
        "Sürüş Görüş Sistemleri": ["ŞAHAN", "AVCI", "NEBULA", "TOYGAR", "MANTIS", "MUCİZE"],
        "Gimbal": ["ALAGÖZ"],
        "Keşif ve Gözetleme Sistemleri": ["YALMAN-150PT", "YALMAN-660PT", "YALMAN-1100PT", "DGV"],
        "Elektro-Optik Nişangahlar": ["DELTA300"],
        "Radar Sistemleri": ["BARKIN-2D", "KURSAD-20A"],
        "Optik Haberleşme Sistemi": ["RAYPATH"]
    }

    for kategori, urunler in kategoriler_ve_urunler.items():
        with st.expander(kategori):
            for urun_adi in urunler:
                st.markdown(
                    f'<a href="?product={urun_adi}" target="_self" class="sidebar-link-btn">{urun_adi}</a>', 
                    unsafe_allow_html=True
                )
                
    st.markdown("---")
    st.caption("© 2026 Lasersan Savunma Sanayii")

if urun_secimi:
    resim_yolu = f"{urun_secimi}.png"
    pdf_yolu = f"urunpdf/{urun_secimi}.pdf"

    if not st.session_state.show_pdf:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col_bosluk1, col_main, col_bosluk2 = st.columns([1, 1, 1]) 
        
        with col_main:
            st.markdown(f"<div class='prod-title-large'>{urun_secimi}</div>", unsafe_allow_html=True)
            
            if os.path.exists(resim_yolu):
                st.image(resim_yolu, use_container_width=True)
            else:
                st.info(f"📷 Lütfen ana dizine '{urun_secimi}.png' adında bir ürün fotoğrafı ekleyin.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(f"📄 Broşürü Görüntüle", use_container_width=True):
                st.session_state.show_pdf = True
                st.rerun()

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col_sol, col_sag = st.columns([1, 2.2]) 
        
        with col_sol:
            st.markdown("""<style>[data-testid="column"]:nth-child(1) [data-testid="stImage"] img { max-height: 250px !important; }</style>""", unsafe_allow_html=True)
            
            st.markdown(f"<div class='prod-title-small'>{urun_secimi}</div>", unsafe_allow_html=True)
            
            if os.path.exists(resim_yolu):
                st.image(resim_yolu, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("❌ Broşürü Kapat", type="primary", use_container_width=True):
                st.session_state.show_pdf = False
                st.rerun()

            if os.path.exists(pdf_yolu):
                with open(pdf_yolu, "rb") as f:
                    pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
                js_code = f"""
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                    body {{ margin: 0; padding: 0; background: transparent; display: flex; justify-content: center; }}
                    .tab-btn {{
                        background-color: #334155; color: #ffffff; padding: 10px; border-radius: 8px;
                        font-weight: 350; font-size: 1.1rem; border: none; cursor: pointer; width: 100%;
                        font-family: 'Segoe UI', sans-serif; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                        margin-top: 5px;
                    }}
                    .tab-btn:hover {{ background-color: #1e293b; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.4); }}
                </style>
                </head>
                <body>
                    <button onclick="openPdf()" class="tab-btn"> Yeni Sekmede Aç</button>
                    <script>
                        function openPdf() {{
                            var b64Data = '{pdf_base64}';
                            var byteCharacters = atob(b64Data);
                            var byteNumbers = new Array(byteCharacters.length);
                            for (var i = 0; i < byteCharacters.length; i++) {{ byteNumbers[i] = byteCharacters.charCodeAt(i); }}
                            var byteArray = new Uint8Array(byteNumbers);
                            var file = new Blob([byteArray], {{ type: 'application/pdf' }});
                            var fileURL = URL.createObjectURL(file);
                            window.open(fileURL, '_blank');
                        }}
                    </script>
                </body>
                </html>
                """
                components.html(js_code, height=60)

        with col_sag:
            if os.path.exists(pdf_yolu):
                
                st.markdown(f"<div class='fancy-pdf-header'><span>{urun_secimi}</span> ÜRÜN BROŞÜRÜ</div>", unsafe_allow_html=True)
                #  Native Iframe 
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}#toolbar=0&navpanes=0&scrollbar=0" width="100%" height="800" type="application/pdf" style="border-radius: 12px; border: 2px solid #107F93; box-shadow: 0 8px 25px rgba(0,0,0,0.6);"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.warning(f"📄 Lütfen 'urunpdf' klasörüne '{urun_secimi}.pdf' adlı broşür dosyasını ekleyin.")

# CHATBOT 
else:
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        logo_yolu = "logo3.png" if os.path.exists("logo3.png") else "logo.png"
        if os.path.exists(logo_yolu):
            st.image(logo_yolu, use_container_width=True)

    st.markdown(
        "<h1 class='main-title'>"
        "<span style='color: #003B70;'>Laser</span>"
        "<span style='color: #C8102E;'>san Akıllı Asistan</span>"
        "</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("<p class='sub-title'>Tüm şirket cihazları hakkında anında ve güvenilir bilgi alın.</p>", unsafe_allow_html=True)

    @st.cache_resource 
    def sistemi_hazirla():
        loader = PyPDFDirectoryLoader("urunpdf/")
        ham_belgeler = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        belgeler = text_splitter.split_documents(ham_belgeler)
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_documents(documents=belgeler, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        llm = OllamaLLM(model="qwen2.5", temperature=0)
        sistem_kurallari = (
            "Sen Lasersan şirketinin resmi yapay zeka asistanısın. Görevin SADECE sana verilen 'BROŞÜR BİLGİLERİ' metnine dayanarak cevap vermektir.\n\n"
            "KATI KURALLAR:\n"
            "1. CİHAZ SORULARI: Kullanıcı bir cihazın özelliğini sorduğunda SADECE 'BROŞÜR BİLGİLERİ' kısmında o bilgi açıkça yazıyorsa cevap ver.\n"
            "2. BİLGİ YOKSA NE YAPACAKSIN: Eğer kullanıcının sorduğu detay broşür metninde YOKSA, KESİNLİKLE tahmin yürütme. Sadece şunu söyle: 'Bu cihazın broşüründe sorduğunuz özellik hakkında bir bilgi bulunmamaktadır.'\n"
            "3. ASLA UYDURMA: Kendiliğinden değer uydurmak (halüsinasyon) veya dışarıdan bilgi eklemek kesinlikle yasaktır.\n"
            "4. ÜRÜN LİSTESİ (EZBERLE): Şirketimizin ürettiği cihazlar şunlardır: AURA, FOCUS, ODAK, TUNAY, NOXIS, TALOS, KOZGU, ŞAHAN, AVCI, NEBULA, TOYGAR, MANTIS, MUCİZE, ALAGÖZ, YALMAN-150PT, YALMAN-660PT, YALMAN-1100PT, DGV, DELTA300, BARKIN-2D, KURSAD-20A, RAYPATH. Eğer kullanıcı ürünleri listele derse KESİNLİKLE broşüre bakma, SADECE bu listedeki isimleri yaz.\n"
            "5. RAKİP KURALI: Aselsan, Roketsan vb. firmalar sorulursa: 'Ben Lasersan'ın resmi asistanıyım ve diğer firmalar hakkında yorum yapmam.' de ve konuyu kapat.\n"
            "6. GÜVENLİK SINIRI: Alakasız konulara cevap verme.\n"
        )
        prompt = ChatPromptTemplate.from_template(
            "{sistem_kurallari}\n\n"
            "--- SOHBET GEÇMİŞİ ---\n{chat_history}\n\n"
            "--- BROŞÜR BİLGİLERİ ---\n{context}\n\n"
            "MÜŞTERİ SORUSU: {input}"
        )
        def belgeleri_birlestir(belgeler):
            return "\n\n---\n\n".join(belge.page_content for belge in belgeler)
        rag_zinciri = (
            {
                "context": itemgetter("arama_metni") | retriever | belgeleri_birlestir,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
                "sistem_kurallari": lambda x: sistem_kurallari
            }
            | prompt
            | llm
            | StrOutputParser()
        ) 
        return rag_zinciri

    zincir = sistemi_hazirla()

    if "mesajlar" not in st.session_state:
        st.session_state.mesajlar = []

    if len(st.session_state.mesajlar) == 0:
        st.markdown("<p class='welcome-text'>Bugün size nasıl yardımcı olabilirim?</p>", unsafe_allow_html=True)

    for mesaj in st.session_state.mesajlar:
        with st.chat_message(mesaj["rol"]):
            st.markdown(mesaj["icerik"])

    soru = st.chat_input("Lasersan cihazlarıyla ilgili bir soru sorun...")

    if soru:
        with st.chat_message("user"):
            st.markdown(soru)
        
        with st.chat_message("assistant"):
            cevap_alani = st.empty()
            cevap_alani.markdown("Düşünüyor ve broşürleri inceliyor...")
            gecmis_metni = ""
            for m in st.session_state.mesajlar[-4:]:
                kim = "Müşteri" if m["rol"] == "user" else "Asistan"
                gecmis_metni += f"{kim}: {m['icerik']}\n"
            arama_metni = soru
            if len(st.session_state.mesajlar) > 0:
                onceki_soru = st.session_state.mesajlar[-1]["icerik"]
                arama_metni = f"{onceki_soru} {soru}"
            cevap = zincir.invoke({
                "input": soru, 
                "arama_metni": arama_metni,
                "chat_history": gecmis_metni
            })
            cevap_alani.markdown(cevap)
            
        st.session_state.mesajlar.append({"rol": "user", "icerik": soru})
        st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})