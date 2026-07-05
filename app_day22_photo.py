import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io

# ========== 页面配置 ==========
st.set_page_config(page_title="拍照写文案", page_icon="📸", layout="wide")

# ========== 修复1：接口地址改回正确的 /v1 ==========
# 注意：密钥名称 DEEPSEEK_API_KEY 必须和你Streamlit后台「秘密」里的名称完全一致
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)

# ========== 修复2：新增图片自动压缩，解决大图报错 ==========
def encode_image(uploaded_file, max_size=1024):
    """
    自动压缩图片：最长边不超过1024像素，统一转JPG格式
    解决PNG大图、手机原图体积过大导致的接口报错
    """
    img = Image.open(uploaded_file)
    
    # 等比例缩小
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    
    # 统一转RGB+JPG，消除PNG透明通道兼容问题
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ========== 修复3：增加错误捕获，出错显示具体原因 ==========
def photo_to_post(image_base64, style, shop_name):
    """VL2：识图 + 写文案，一次调用"""
    try:
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
    except Exception as e:
        # 出错时返回完整错误信息，方便排查
        return f"❌ 生成失败\n错误原因：{str(e)}"

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
