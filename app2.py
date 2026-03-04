import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Lasersan AI", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")

custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stChatInputContainer > div {
        border-radius: 24px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        padding-left: 10px;
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10a37f !important;
        color: white !important;
    }
    
    .main-title {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: -webkit-linear-gradient(45deg, #10a37f, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 10px;
    }
    
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>Lasersan Akıllı Asistan</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Tüm şirket cihazları hakkında anında ve güvenilir bilgi alın.</p>", unsafe_allow_html=True)

@st.cache_resource #baştan return 
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
        "Sen Lasersan şirketinin resmi yapay zeka asistanısın. Görevin sadece şirketimizin cihazları hakkında bilgi vermektir.\n\n"
        "GÖREVLERİN VE KURALLARIN:\n"
        "1. ÜRÜN LİSTESİ (EZBERLE): Şirketimizin ürettiği 24 adet cihaz şunlardır: ŞAHAN, NOXIS, NEBULA, TUNAY, AURA, FOCUS, ODAK, TALOS-L, KOZGU, AVCI, TOYGAR, BARBAROS, MIRACLE, ALAGÖZ, YALMAN-150PT, DELTA-180PT, DELTA-225PT, YALMAN-660PT, YALMAN-1100PT, DELTA-1100PT, BARKIN-2D, KURSAD-20A, RAYPATH. (NOT: Varsa 24. cihazı buraya ekleyin). Eğer kullanıcı 'tüm ürünleri say', 'hangi ürünleriniz var', 'cihazları listele' derse KESİNLİKLE broşüre bakma, doğrudan ve SADECE bu listedeki 24 ismi alt alta yaz. Asla isim uydurma (Sen Şahane vb. yapma).\n"
        "2. CİHAZ SORULARI: Kullanıcı bir cihazı 'anlat', 'tanıt', 'nedir' veya 'özelliklerini ver' derse, aslında o cihazın genel/teknik detaylarını istiyordur. DOĞRUDAN 'Broşür Bilgileri' kısmına bak, cihazı bul ve detayları listele. Sana verilen metni dikkatlice okumadan 'broşürde yok' deme.\n"
        "3. RAKİP KURALI: Aselsan, Roketsan, Baykar vb. firmalar sorulursa: 'Ben Lasersan'ın resmi asistanıyım ve diğer firmalar hakkında yorum yapmam.' de ve konuyu kapat.\n"
        "4. GÜVENLİK SINIRI: Alakasız konulara (yemek, matematik vb.) cevap verme. 'Size sadece şirketimiz hakkında yardımcı olabilirim' de.\n"
        "5. Broşürde olmayan hiçbir bilgiyi uydurma.\n\n"
        "Broşür Bilgileri:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", sistem_kurallari),
        ("human", "{input}"),
    ])

    def belgeleri_birlestir(belgeler):
        return "\n\n---\n\n".join(belge.page_content for belge in belgeler)
    
    rag_zinciri = (
        {"context": retriever | belgeleri_birlestir, "input": RunnablePassthrough()}
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
    st.session_state.mesajlar.append({"rol": "user", "icerik": soru})

    with st.chat_message("assistant"):
        cevap_alani = st.empty()
        cevap_alani.markdown("Broşürler inceleniyor...")

        cevap = zincir.invoke(soru)
        cevap_alani.markdown(cevap)
        
    st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})