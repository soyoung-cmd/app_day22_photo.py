import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image
import io
import base64
import json
import time

st.set_page_config(page_title="微信风格客服", page_icon="💬", layout="wide")

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

MODEL_NAME = "ep-20260705180241-s57gl"

# ==================== 知识库初始化 ====================
@st.cache_resource
def get_embedding():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="shibing624/text2vec-base-chinese"
    )

@st.cache_resource
def init_kb():
    lines = [
        "营业时间：10:00-23:00", "预约电话：13812345678",
        "3个包间，提前1天预订", "人均120元", "招牌菜：酸菜鱼、麻辣香锅、蒜蓉小龙虾",
        "会员充值500送50，1000送150", "每周二会员日8折", "生日当天凭身份证送招牌菜一份"
    ]
    ef = get_embedding()
    cc = chromadb.Client()
    try:
        cc.delete_collection("wx_kb")
    except:
        pass
    col = cc.create_collection(name="wx_kb", embedding_function=ef)
    for i, l in enumerate(lines):
        col.add(documents=[l], ids=[str(i)])
    return col, lines

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
    {"type":"function","function":{"name":"query_order","description":"查订单","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}},
    {"type":"function","function":{"name":"refund_order","description":"退款","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}}
]

def exec_tool(name, args):
    if name == "query_order": return query_order(args["order_id"])
    if name == "refund_order": return refund_order(args["order_id"])
    return "未知操作"

# ==================== 图片处理 ====================
def encode_image(uploaded_file, max_size=1024):
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ==================== 核心函数 ====================
def smart_reply(user_input, history, kb_collection, image_b64=None):
    """智能回复：判断意图，调用对应功能"""
    
    # 第一步：检索知识库
    results = kb_collection.query(query_texts=[user_input], n_results=2)
    docs = results['documents'][0]
    knowledge = "\n".join(docs)
    
    # 第二步：构建消息
    system_prompt = f"""你是一个亲切的店铺微信客服。
参考以下店铺知识回答顾客问题：
{knowledge}

规则：
1. 语气像朋友聊天，用"亲""呢""哦"等语气词
2. 回答简洁，30-50字
3. 如果顾客想查订单或退款，使用工具
4. 如果顾客发图片，描述并给出建议"""
    
    messages = [{"role": "system", "content": system_prompt}] + history
    
    # 如果有图片
    if image_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_input or "请分析这张图片"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = exec_tool(tool_call.function.name, args)
            
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            
            final = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            return final.choices[0].message.content, {"tool": tool_call.function.name, "result": result}
        
        return msg.content, None
    
    except Exception as e:
        return f"❌ 抱歉，系统出了点问题：{str(e)}", None

# ==================== 初始化状态 ====================
if "kb_collection" not in st.session_state:
    st.session_state.kb_collection, st.session_state.kb_lines = init_kb()

if "wx_messages" not in st.session_state:
    st.session_state.wx_messages = [
        {"role": "assistant", "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？\n\n你可以问我：\n• 营业时间、菜单价格\n• 订单到哪里了\n• 发张照片我帮你写文案"}
    ]

if "wx_history" not in st.session_state:
    st.session_state.wx_history = []

# ==================== 微信风格UI ====================
# 顶部仿微信标题栏
st.markdown("""
<div style="background-color:#07C160; padding:10px; border-radius:10px 10px 0 0; text-align:center;">
    <h3 style="color:white; margin:0;">💬 店铺客服</h3>
    <small style="color:#E8F5E9;">在线 · 秒回</small>
</div>
""", unsafe_allow_html=True)

# 聊天区域
chat_container = st.container()

with chat_container:
    for msg in st.session_state.wx_messages:
        if msg["role"] == "user":
            # 用户消息：右侧绿色气泡
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                <div style="background-color:#95EC69; padding:10px 15px; border-radius:15px 5px 15px 15px; max-width:70%;">
                    <small>{msg["content"]}</small>
                </div>
                <div style="width:35px; height:35px; background-color:#07C160; border-radius:50%; margin-left:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:14px;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI消息：左侧白色气泡
            content_display = msg["content"]
            if msg.get("tool_log"):
                content_display += f"\n\n🔧 [{msg['tool_log']['tool']}]"
            
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
                <div style="width:35px; height:35px; background-color:#07C160; border-radius:50%; margin-right:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:14px;">🤖</div>
                <div style="background-color:white; padding:10px 15px; border-radius:5px 15px 15px 15px; max-width:70%; border:1px solid #E0E0E0;">
                    <small>{content_display}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# 底部输入栏
st.divider()
col1, col2, col3 = st.columns([5, 1, 1])

with col1:
    user_input = st.chat_input("输入消息...", key="wx_input")

with col2:
    upload_img = st.file_uploader("📷", type=["jpg","jpeg","png"], label_visibility="collapsed")

with col3:
    if st.button("🔄", help="清空对话"):
        st.session_state.wx_messages = [
            {"role": "assistant", "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？"}
        ]
        st.session_state.wx_history = []
        st.rerun()

# ==================== 处理输入 ====================
if user_input:
    # 添加用户消息
    st.session_state.wx_messages.append({"role": "user", "content": user_input})
    
    # 获取AI回复
    with st.spinner(""):
        reply, tool_log = smart_reply(
            user_input,
            st.session_state.wx_history,
            st.session_state.kb_collection
        )
    
    msg_data = {"role": "assistant", "content": reply}
    if tool_log:
        msg_data["tool_log"] = tool_log
    
    st.session_state.wx_messages.append(msg_data)
    st.session_state.wx_history.append({"role": "user", "content": user_input})
    st.session_state.wx_history.append({"role": "assistant", "content": reply})
    st.rerun()

if upload_img:
    img_b64 = encode_image(upload_img)
    st.session_state.wx_messages.append({"role": "user", "content": "[图片]"})
    
    with st.spinner(""):
        reply, tool_log = smart_reply(
            "请分析这张图片，如果是菜品就生成朋友圈文案",
            st.session_state.wx_history,
            st.session_state.kb_collection,
            img_b64
        )
    
    msg_data = {"role": "assistant", "content": reply}
    if tool_log:
        msg_data["tool_log"] = tool_log
    
    st.session_state.wx_messages.append(msg_data)
    st.rerun()
