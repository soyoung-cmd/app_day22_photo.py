import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import json

st.set_page_config(page_title="微信风格客服", page_icon="💬", layout="wide")

# ========== 模型配置 ==========
client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# 纯文字模型：对话、知识库、工具调用（替换成你的DeepSeek-V4-flash接入点ID）
TEXT_MODEL = "ep-你的纯文字模型接入点ID"
# 视觉模型：图片识别、写文案（不变）
VISION_MODEL = "ep-20260705180241-s57gl"

# ==================== 店铺知识库 ====================
KB_DATA = [
    "营业时间：10:00-23:00",
    "预约电话：13812345678",
    "3个包间，提前1天预订",
    "人均120元",
    "招牌菜：酸菜鱼、麻辣香锅、蒜蓉小龙虾",
    "会员充值500送50，1000送150",
    "每周二会员日8折",
    "生日当天凭身份证送招牌菜一份"
]

def search_kb(query, top_k=2):
    scores = []
    for item in KB_DATA:
        score = sum(1 for word in query if word in item)
        scores.append((score, item))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in scores[:top_k] if score > 0]

# ==================== 订单系统 ====================
orders_db = {
    "001": {"customer": "张三", "item": "酸菜鱼", "status": "配送中", "phone": "13900001111"},
    "002": {"customer": "李四", "item": "麻辣香锅", "status": "已签收", "phone": "13900002222"},
}

def query_order(oid):
    if oid in orders_db:
        o = orders_db[oid]
        return f"订单{oid}：{o['customer']}，{o['item']}，状态：{o['status']}"
    return "未找到该订单"

def refund_order(oid):
    if oid in orders_db:
        orders_db[oid]["status"] = "已退款"
        return f"订单{oid}已退款成功"
    return "退款失败"

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单的物流状态，需要订单号",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单编号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "给指定订单办理退款",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "要退款的订单号"}},
                "required": ["order_id"]
            }
        }
    }
]

def exec_tool(name, args):
    try:
        if name == "query_order":
            return query_order(args["order_id"])
        if name == "refund_order":
            return refund_order(args["order_id"])
        return "未知操作"
    except Exception as e:
        return f"工具执行失败：{str(e)}"

# ==================== 图片处理（已修复语法错误） ====================
def encode_image(uploaded_file, max_size=1024):
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_image(image_b64, prompt):
    """调用视觉模型分析图片，返回纯文本结果"""
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"图片识别失败：{str(e)}"

# ==================== 核心对话函数 ====================
def chat_with_agent(user_text, history):
    kb_docs = search_kb(user_text)
    knowledge = "\n".join(kb_docs) if kb_docs else "无相关店铺信息"

    system_prompt = f"""你是亲切的店铺微信客服，说话像朋友聊天，多用"亲""呢""哦"，回答简洁30-50字。
参考店铺知识回答：
{knowledge}

规则：
1. 用户问订单、退款，自动调用对应工具
2. 不知道的信息如实告知，不要编造
3. 语气友好接地气，符合实体店客服风格"""

    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {"order_id": ""}
            
            result = exec_tool(tool_call.function.name, args)
            
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            final_resp = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            final_content = final_resp.choices[0].message.content
            return final_content, messages[1:], {"tool": tool_call.function.name, "result": result}
        
        return msg.content, messages[1:], None
    
    except Exception as e:
        return f"❌ 系统异常：{str(e)}", history, None

# ==================== 初始化状态 ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？\n\n你可以问我：\n• 营业时间、菜单价格\n• 订单到哪里了\n• 发张照片我帮你写文案"}
    ]
# 防止上传图片无限循环重复处理
if "last_processed_img" not in st.session_state:
    st.session_state.last_processed_img = None

# ==================== 微信风格UI ====================
st.markdown("""
<div style="background-color:#07C160; padding:10px; border-radius:10px 10px 0 0; text-align:center;">
    <h3 style="color:white; margin:0;">💬 店铺客服</h3>
    <small style="color:#E8F5E9;">在线 · 秒回</small>
</div>
""", unsafe_allow_html=True)

# 聊天区域
chat_container = st.container()
with chat_container:
    for msg in st.session_state.ui_messages:
        if msg["role"] == "user":
            content_html = msg["content"]
            if msg.get("image"):
                content_html = f'<img src="{msg["image"]}" style="max-width:200px; border-radius:8px;"/><br/>' + content_html
            
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                <div style="background-color:#95EC69; padding:10px 15px; border-radius:15px 5px 15px 15px; max-width:70%; word-break:break-all;">
                    <small>{content_html}</small>
                </div>
                <div style="width:35px; height:35px; background-color:#07C160; border-radius:50%; margin-left:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:14px; flex-shrink:0;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content_display = msg["content"]
            if msg.get("tool_log"):
                content_display += f"\n\n🔧 已执行：{msg['tool_log']['tool']}"
            
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
                <div style="width:35px; height:35px; background-color:#07C160; border-radius:50%; margin-right:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:14px; flex-shrink:0;">🤖</div>
                <div style="background-color:white; padding:10px 15px; border-radius:5px 15px 15px 15px; max-width:70%; border:1px solid #E0E0E0; word-break:break-all; white-space:pre-wrap;">
                    <small>{content_display}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# 底部操作栏
st.divider()
col_img, col_input, col_clear = st.columns([1, 6, 1])

with col_img:
    upload_img = st.file_uploader("📷", type=["jpg","jpeg","png"], label_visibility="collapsed", key="img_uploader")

with col_input:
    user_input = st.chat_input("输入消息...", key="wx_input")

with col_clear:
    if st.button("🔄", help="清空对话"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？"}
        ]
        st.session_state.last_processed_img = None
        st.rerun()

# ==================== 处理输入 ====================
# 处理文字消息
if user_input:
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    
    with st.spinner(""):
        reply, new_history, tool_log = chat_with_agent(
            user_input,
            st.session_state.chat_history
        )
    
    st.session_state.chat_history = new_history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
    ai_msg = {"role": "assistant", "content": reply}
    if tool_log:
        ai_msg["tool_log"] = tool_log
    st.session_state.ui_messages.append(ai_msg)
    st.rerun()

# 处理图片消息（防无限循环）
if upload_img:
    file_unique_id = f"{upload_img.name}_{upload_img.size}"
    if file_unique_id != st.session_state.last_processed_img:
        img_b64 = encode_image(upload_img)
        img_data_url = f"data:image/jpeg;base64,{img_b64}"
        
        st.session_state.ui_messages.append({
            "role": "user",
            "content": "帮我看看这张图，写个朋友圈文案",
            "image": img_data_url
        })
        
        with st.spinner("正在识别图片..."):
            image_result = analyze_image(
                img_b64,
                "识别这道菜品，描述菜品特点，然后写一条适合实体店发的朋友圈文案，40-80字，带emoji"
            )
            
            reply, new_history, tool_log = chat_with_agent(
                f"用户发了一张菜品图片，识别结果如下：\n{image_result}\n请整理成友好的客服回复",
                st.session_state.chat_history
            )
        
        st.session_state.chat_history = new_history
        st.session_state.chat_history.append({"role": "user", "content": f"[图片] {image_result}"})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        
        ai_msg = {"role": "assistant", "content": reply}
        if tool_log:
            ai_msg["tool_log"] = tool_log
        st.session_state.ui_messages.append(ai_msg)
        
        st.session_state.last_processed_img = file_unique_id
        st.rerun()
