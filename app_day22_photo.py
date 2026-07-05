import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(page_title="拍照写文案", page_icon="📸", layout="wide")

client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/vl2"
)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def photo_to_post(image_base64, style, shop_name):
    """VL2：识图 + 写文案，一次调用"""
    response = client.chat.completions.create(
        model="deepseek-vl2",
        messages=[
            {
                "role": "system",
                "content": f"""你是餐饮营销专家。
看到菜品照片后，请完成：
1. 描述菜品：菜名、食材、颜色、口感（30字内）
2. 为{shop_name}写{style}风格朋友圈文案（40-80字，带emoji，结尾引导互动）

输出格式：
【识别】
（描述）
【文案】
（文案内容）"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析菜品并生成文案"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def parse_result(raw):
    if "【文案】" in raw:
        parts = raw.split("【文案】")
        desc = parts[0].replace("【识别】", "").strip()
        post = parts[1].strip()
        return desc, post
    return raw, ""

# ==================== UI ====================
st.title("📸 拍照写文案 · DeepSeek VL2")
st.markdown("拍菜品照片，AI自动识别并生成朋友圈文案")

col1, col2 = st.columns([1, 1])

with col1:
    style = st.selectbox("文案风格", ["小红书种草风 🌿", "幽默搞笑风 😂", "高端精致风 ✨", "温情故事风 💝"])
    shop_name = st.text_input("店铺名称", value="老成都火锅店")
    uploaded_file = st.file_uploader("上传菜品照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="上传的图片", use_container_width=True)
    
    btn = st.button("🚀 生成文案", type="primary", use_container_width=True)

with col2:
    st.subheader("📝 生成结果")
    
    if btn and uploaded_file:
        with st.spinner("VL2 识别并生成中..."):
            img_b64 = encode_image(uploaded_file)
            raw = photo_to_post(img_b64, style, shop_name)
            desc, post = parse_result(raw)
            
            st.markdown("**🔍 AI识别：**")
            st.info(desc)
            st.markdown("**📝 朋友圈文案：**")
            st.success(post)
            st.code(post, language="text")
            
            with st.expander("🔧 原始返回"):
                st.text(raw)

# 摄像头拍照
st.divider()
st.subheader("📸 或用摄像头直接拍")
camera_photo = st.camera_input("对着菜品拍一张")

if camera_photo:
    with st.spinner("处理中..."):
        img_b64 = base64.b64encode(camera_photo.getvalue()).decode('utf-8')
        raw = photo_to_post(img_b64, "小红书种草风 🌿", "我的店铺")
        desc, post = parse_result(raw)
        st.markdown("**🔍 识别：**")
        st.info(desc)
        st.markdown("**📝 文案：**")
        st.success(post)
