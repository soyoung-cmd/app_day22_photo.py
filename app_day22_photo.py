import streamlit as st
import base64
import requests
from PIL import Image
import io

# ========== 页面配置 ==========
st.set_page_config(page_title="拍照写文案", page_icon="📸", layout="wide")

# ========== 读取密钥 ==========
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("⚠️ 请在后台配置 DEEPSEEK_API_KEY")
    st.stop()

API_KEY = st.secrets["DEEPSEEK_API_KEY"]
# VL2 专用接口地址（注意：不是通用的 /v1/chat/completions）
VL2_API_URL = "https://api.deepseek.com/vl2"

# ========== 图片自动压缩（解决大图报错）==========
def encode_image(uploaded_file, max_size=1024):
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ========== 调用 DeepSeek-VL2 专用接口 ==========
def photo_to_post(image_base64, style, shop_name):
    # 构造完整的提示词
    prompt = f"""你是餐饮营销专家。
看到菜品照片后，请完成：
1. 描述菜品：菜名、食材、颜色、口感（30字内）
2. 为{shop_name}写{style}风格朋友圈文案（40-80字，带emoji，结尾引导互动）

输出格式：
【识别】
（描述）
【文案】
（文案内容）"""

    try:
        # 官方 VL2 专用请求格式
        payload = {
            "model": "deepseek-vl2",
            "text": prompt,
            "image_url": f"data:image/jpeg;base64,{image_base64}",
            "temperature": 0.7,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(VL2_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 提取返回结果
        return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"❌ 生成失败\n错误原因：{str(e)}\n响应详情：{response.text if 'response' in dir() else '无'}"

# ========== 结果解析 ==========
def parse_result(raw):
    if "【文案】" in raw:
        parts = raw.split("【文案】")
        desc = parts[0].replace("【识别】", "").strip()
        post = parts[1].strip()
        return desc, post
    return raw, ""

# ==================== UI 界面 ====================
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
            
            if raw.startswith("❌"):
                st.error(raw)
            else:
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
        img_b64 = encode_image(camera_photo)
        raw = photo_to_post(img_b64, "小红书种草风 🌿", "我的店铺")
        desc, post = parse_result(raw)
        
        if raw.startswith("❌"):
            st.error(raw)
        else:
            st.markdown("**🔍 识别：**")
            st.info(desc)
            st.markdown("**📝 文案：**")
            st.success(post)
