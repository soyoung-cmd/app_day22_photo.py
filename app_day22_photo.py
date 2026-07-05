import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io

st.set_page_config(page_title="发型推荐", page_icon="💇", layout="wide")

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

MODEL_NAME = "ep-20260705180241-s57gl"

def encode_image(uploaded_file, max_size=1024):
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_and_recommend(image_base64, gender, style_pref):
    prompt = f"""你是资深发型设计师。
看到自拍照片后，请完成：
1. 分析脸型、当前发型、肤色（50字内）
2. 为{gender}性推荐3款{style_pref}发型

输出格式：
【面部分析】
（分析内容）
【发型推荐】
1. 发型名：适合理由、打理难度、适合场合
2. 发型名：适合理由、打理难度、适合场合
3. 发型名：适合理由、打理难度、适合场合"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 分析失败\n错误原因：{str(e)}"

def parse_result(raw):
    if raw.startswith("❌"):
        return raw, ""
    if "【发型推荐】" in raw:
        parts = raw.split("【发型推荐】")
        analysis = parts[0].replace("【面部分析】", "").strip()
        recommend = parts[1].strip()
        return analysis, recommend
    return raw, ""

st.title("💇 AI发型推荐")
st.markdown("上传自拍，AI分析脸型并推荐发型")

col1, col2 = st.columns([1, 1])

with col1:
    gender = st.radio("性别", ["女", "男"], horizontal=True)
    style_pref = st.selectbox("风格偏好", ["韩系清新", "日系甜美", "欧美时尚", "简约通勤", "个性潮酷"])
    uploaded_file = st.file_uploader("上传自拍照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="你的自拍", use_container_width=True)
    
    btn = st.button("🔍 分析并推荐", type="primary", use_container_width=True)

with col2:
    if btn and uploaded_file:
        with st.spinner("AI分析中..."):
            img_b64 = encode_image(uploaded_file)
            raw = analyze_and_recommend(img_b64, gender, style_pref)
            
            if raw.startswith("❌"):
                st.error(raw)
            else:
                analysis, recommend = parse_result(raw)
                st.markdown("**🔍 面部分析：**")
                st.info(analysis)
                st.markdown("**💇 发型推荐：**")
                st.success(recommend)

st.divider()
st.caption("💡 正面、光线充足的自拍效果最好")
st.caption("⚠️ 图片仅用于本次分析，不会被存储")
