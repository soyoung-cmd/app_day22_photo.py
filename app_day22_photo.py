import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import base64
from PIL import Image
import io

st.set_page_config(page_title="全能助手", page_icon="🤖", layout="wide")

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

MODEL_NAME = "ep-20260705180241-s57gl"

@st.cache_resource
def get_embedding():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="shibing624/text2vec-base-chinese"
    )

def init_knowledge(file_content=None):
    if file_content:
        lines = [l.strip() for l in file_content.split("\n") if l.strip()]
    else:
        lines = [
            "老成都火锅店营业时间：10:00-23:00",
            "预约电话：13812345678",
            "有3个包间，需提前1天预订",
            "人均消费：120元",
            "招牌菜：酸菜鱼、麻辣香锅、蒜蓉小龙虾"
        ]
    
    ef = get_embedding()
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection("shop_kb")
    except:
        pass
    
    collection = chroma_client.create_collection(name="shop_kb", embedding_function=ef)
    for i, line in enumerate(lines):
        collection.add(documents=[line], ids=[str(i)])
    
    return collection, lines

def encode_image(uploaded_file, max_size=1024):
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def vision_chat(image_base64, question):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 识别失败：{str(e)}"

def rag_chat(question, collection):
    results = collection.query(query_texts=[question], n_results=2)
    docs = results['documents'][0]
    knowledge = "\n".join(docs)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"你是店铺客服。根据以下知识回答，亲切简洁：\n{knowledge}"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content, docs
    except Exception as e:
        return f"❌ 问答失败：{str(e)}", docs

if "collection" not in st.session_state:
    st.session_state.collection, st.session_state.lines = init_knowledge()

st.title("🤖 店小秘AI · 全能助手")
st.markdown("拍照问答 + 知识库客服，一个模型全搞定")

tab1, tab2, tab3 = st.tabs(["📸 拍照问答", "💬 知识库客服", "📂 知识库管理"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        question = st.text_input("你想问什么？", placeholder="例如：这道菜叫什么？怎么做的？")
        uploaded = st.file_uploader("上传图片", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, use_container_width=True)
        btn = st.button("🔍 看图回答", type="primary", use_container_width=True)
    with col2:
        if btn and uploaded and question:
            with st.spinner("AI分析中..."):
                answer = vision_chat(encode_image(uploaded), question)
                if answer.startswith("❌"):
                    st.error(answer)
                else:
                    st.markdown("**🤖 回答：**")
                    st.success(answer)

with tab2:
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []
    
    for msg in st.session_state.chat_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if user_input := st.chat_input("输入顾客问题..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("检索知识库..."):
            reply, docs = rag_chat(user_input, st.session_state.collection)
        
        with st.chat_message("assistant"):
            if reply.startswith("❌"):
                st.error(reply)
            else:
                st.markdown(reply)
                with st.expander("📚 参考知识"):
                    for d in docs:
                        st.text(f"• {d}")
        
        st.session_state.chat_msgs.append({"role":"user","content":user_input})
        st.session_state.chat_msgs.append({"role":"assistant","content":reply})

with tab3:
    st.subheader("📂 上传知识文件")
    uploaded_kb = st.file_uploader("选择txt文件", type="txt")
    if uploaded_kb:
        content = uploaded_kb.read().decode("utf-8")
        st.session_state.collection, st.session_state.lines = init_knowledge(content)
        st.success(f"知识库已更新，共 {len(st.session_state.lines)} 条")
        with st.expander("预览"):
            for i, line in enumerate(st.session_state.lines):
                st.text(f"{i+1}. {line}")
