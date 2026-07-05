import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import json
import re

# ====================== 页面基础配置 ======================
st.set_page_config(page_title="门店专属客服", page_icon="🍲", layout="wide")

# ====================== API客户端初始化 ======================
# 校验密钥配置
if "DOUBAO_API_KEY" not in st.secrets:
    st.error("❌ 后台未配置API密钥，请在secrets.toml填写 DOUBAO_API_KEY = '你的sk密钥'")
    st.stop()

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# 模型接入ID
TEXT_MODEL = "ep-20260705195949-pr84t"
VISION_MODEL = "ep-20260705180241-s57gl"

# ====================== 店铺静态知识库 ======================
KB_DATA = [
    "营业时间：每日10:00-23:00，节假日正常营业",
    "预约电话：13812345678，建议提前1天预定包间",
    "门店共3个独立包间，用餐高峰期容易满房",
    "店内人均消费120元，丰俭由人",
    "招牌菜品：酸菜鱼、麻辣香锅、蒜蓉小龙虾",
    "会员充值活动：充500送50，充1000送150，长期有效",
    "每周二会员专属日，全场菜品8折优惠",
    "生日到店凭本人身份证，免费赠送招牌酸菜鱼一份"
]

def search_kb(query, top_k=2):
    """关键词检索门店信息"""
    score_list = []
    for text in KB_DATA:
        match_count = sum(1 for word in query if word in text)
        score_list.append((match_count, text))
    score_list.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in score_list[:top_k] if score > 0]

# ====================== 订单模拟数据库 & 工具函数 ======================
orders_db = {
    "001": {"customer": "张三", "item": "酸菜鱼", "status": "配送中", "phone": "13900001111"},
    "002": {"customer": "李四", "item": "麻辣香锅", "status": "已签收", "phone": "13900002222"},
}

def query_order(order_id: str):
    if order_id in orders_db:
        info = orders_db[order_id]
        return f"订单{order_id}：顾客{info['customer']}，菜品{info['item']}，当前配送状态：{info['status']}"
    return "未查询到该订单，请仔细核对订单编号后重新查询"

def refund_order(order_id: str):
    if order_id in orders_db:
        orders_db[order_id]["status"] = "已退款"
        return f"订单{order_id}退款申请已受理，款项1-3个工作日原路退回"
    return "退款操作失败，不存在该订单编号"

# 工具定义
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询顾客外卖订单配送进度，必须获取用户提供的订单编号",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "用户提供的订单编号"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "为顾客订单办理退款，需要用户提供订单编号",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "需要退款的订单编号"}
                },
                "required": ["order_id"]
            }
        }
    }
]

def execute_tool(tool_name: str, args: dict):
    try:
        if tool_name == "query_order":
            return query_order(args["order_id"])
        elif tool_name == "refund_order":
            return refund_order(args["order_id"])
        else:
            return f"暂不支持该功能：{tool_name}"
    except Exception as e:
        return f"功能执行出错：{str(e)}"

# ====================== 图片处理工具 ======================
def encode_image(upload_file, max_edge=1024):
    img = Image.open(upload_file)
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str

def clean_emoji(text: str) -> str:
    """清除大量杂乱表情，只保留1个统一表情"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    clean = emoji_pattern.sub(r'', text).strip()
    return clean + "😊"

# ====================== 视觉图片识别函数 ======================
def vision_analyze(image_b64: str, prompt_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=600
        )
        raw = resp.choices[0].message.content
        return clean_emoji(raw)
    except Exception as err:
        return f"图片识别失败：{str(err)}"

# ====================== 文字对话核心Agent ======================
def text_agent_chat(user_input: str, history_msg_list: list):
    kb_match_text = "\n".join(search_kb(user_input)) or "暂无相关门店信息"
    # 强约束人性化系统提示词
    system_prompt = f"""你是线下餐饮实体店贴心客服，严格遵守所有规则，禁止敷衍回复：
【硬性规则】
1. 完整解答用户全部问题，回答长度控制40-70字，禁止只发短句；
2. 语气温和接地气，少量使用亲、呢、哦，不要堆砌大量表情；
3. 门店参考信息：
{kb_match_text}
4. 用户询问订单、退款，主动调用对应工具查询；不知道的信息如实告知，不编造；
5. 用户上传菜品图片，详细描述菜品优势，生成适合朋友圈、小红书的营销文案；
6. 顾客索要宣传文案时，文案适合实体店直接复制发社交平台。"""

    msg_list = [{"role": "system", "content": system_prompt}] + history_msg_list
    msg_list.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=msg_list,
            tools=tool_definitions,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=500
        )
        ai_msg = response.choices[0].message

        # 存在工具调用
        if ai_msg.tool_calls:
            tool_call_info = ai_msg.tool_calls[0]
            tool_args = json.loads(tool_call_info.function.arguments)
            tool_result_content = execute_tool(tool_call_info.function.name, tool_args)

            msg_list.append(ai_msg)
            msg_list.append({
                "role": "tool",
                "tool_call_id": tool_call_info.id,
                "content": tool_result_content
            })

            final_response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=msg_list,
                temperature=0.4,
                max_tokens=350
            )
            final_text = clean_emoji(final_response.choices[0].message.content)
            return final_text, msg_list[1:], {"tool": tool_call_info.function.name, "result": tool_result_content}
        else:
            raw_text = ai_msg.content
            clean_text = clean_emoji(raw_text)
            return clean_text, msg_list[1:], None

    except Exception as err:
        return f"❌ 服务暂时异常，请稍后重试：{str(err)}", history_msg_list, None

# ====================== 会话状态初始化 ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": "亲，您好！我是本店专属客服，有任何问题都可以跟我说哦😊\n您可以咨询营业时间、包间预约、会员活动，上传菜品图片我还能帮您写宣传文案！"
        }
    ]

if "last_handle_img_id" not in st.session_state:
    st.session_state.last_handle_img_id = None

# ====================== 页面UI渲染 ======================
st.markdown("""
<div style="background-color:#07C160; padding:12px; border-radius:12px 12px 0 0; text-align:center;">
    <h3 style="color:white; margin:0; font-size:20px;">🍲 门店在线客服</h3>
    <small style="color:#E8F5E9;">实时在线 · 一键生成菜品宣传文案</small>
</div>
""", unsafe_allow_html=True)

# 快捷功能按钮区
st.divider()
col1, col2, col3, col4 = st.columns([1,1,1,1])
quick_text = ""
with col1:
    if st.button("⏰ 营业时间"):
        quick_text = "门店营业时间是多少？"
with col2:
    if st.button("🎁 会员活动"):
        quick_text = "介绍一下门店会员充值优惠"
with col3:
    if st.button("📦 查询订单"):
        quick_text = "帮我查一下订单，需要提供订单编号"
with col4:
    if st.button("📸 菜品文案"):
        quick_text = "我上传菜品图片，帮我写朋友圈宣传文案"

# 快捷按钮自动发送消息
if quick_text:
    st.session_state.ui_messages.append({"role": "user", "content": quick_text})
    with st.spinner("正在为您解答..."):
        reply_text, updated_history, tool_log_data = text_agent_chat(quick_text, st.session_state.chat_history)
    st.session_state.chat_history = updated_history
    st.session_state.chat_history.append({"role": "user", "content": quick_text})
    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
    ai_msg_item = {"role": "assistant", "content": reply_text}
    if tool_log_data:
        ai_msg_item["tool_log"] = tool_log_data
    st.session_state.ui_messages.append(ai_msg_item)
    st.rerun()

# 聊天消息容器
chat_box = st.container(height=520)
with chat_box:
    for idx, single_msg in enumerate(st.session_state.ui_messages):
        if single_msg["role"] == "user":
            display_content = single_msg["content"]
            if "image" in single_msg:
                display_content = f'<img src="{single_msg["image"]}" style="max-width:200px; border-radius:8px;"><br/>' + display_content
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:10px 0;">
                <div style="background:#95EC69; padding:10px 15px; border-radius:15px 5px 15px 15px; max-width:72%; word-break:break-all;">
                    <small>{display_content}</small>
                </div>
                <div style="width:36px;height:36px;background:#07C160;border-radius:50%;margin-left:8px;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            display_content = single_msg["content"]
            if "tool_log" in single_msg:
                display_content += f"\n\n🔧 已执行功能：{single_msg['tool_log']['tool']}"
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin:10px 0;">
                <div style="width:36px;height:36px;background:#07C160;border-radius:50%;margin-right:8px;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;">🤖</div>
                <div style="background:#fff; padding:10px 15px; border:1px solid #e6e6e6; border-radius:5px 15px 15px 15px; max-width:72%; white-space:pre-wrap; word-break:break-all;">
                    <small>{display_content}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # 文案复制按钮（仅AI回复展示）
            st.button(f"📋 复制本条文案", key=f"copy_{idx}", help="一键复制文案发朋友圈")
            st.code(display_content, language="text")

# 底部输入栏
st.divider()
col_upload, col_input, col_reset = st.columns([1, 6, 1])

with col_upload:
    upload_img_file = st.file_uploader("📷 上传菜品", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img_upload")

with col_input:
    user_text_input = st.chat_input("输入您的问题...", key="chat_text_input")

with col_reset:
    if st.button("🔄 清空对话", help="重置全部聊天记录"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "亲，您好！我是本店专属客服，有任何问题都可以跟我说哦😊\n您可以咨询营业时间、包间预约、会员活动，上传菜品图片我还能帮您写宣传文案！"}
        ]
        st.session_state.last_handle_img_id = None
        st.rerun()

# ====================== 文本消息处理 ======================
if user_text_input:
    st.session_state.ui_messages.append({"role": "user", "content": user_text_input})
    with st.spinner("AI正在整理回复..."):
        reply_text, updated_history, tool_log_data = text_agent_chat(user_text_input, st.session_state.chat_history)
    st.session_state.chat_history = updated_history
    st.session_state.chat_history.append({"role": "user", "content": user_text_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
    ai_msg_item = {"role": "assistant", "content": reply_text}
    if tool_log_data:
        ai_msg_item["tool_log"] = tool_log_data
    st.session_state.ui_messages.append(ai_msg_item)
    st.rerun()

# ====================== 图片上传处理（防无限循环） ======================
if upload_img_file:
    current_img_tag = f"{upload_img_file.name}_{upload_img_file.size}"
    if current_img_tag != st.session_state.last_handle_img_id:
        img_b64 = encode_image(upload_img_file)
        img_data_url = f"data:image/jpeg;base64,{img_b64}"
        st.session_state.ui_messages.append({
            "role": "user",
            "content": "帮我分析这道菜品，生成适合实体店发朋友圈的宣传文案",
            "image": img_data_url
        })
        with st.spinner("识别菜品+撰写宣传文案中..."):
            img_analysis_result = vision_analyze(
                img_b64,
                "精准识别图中菜品，描述口感、食材亮点，生成40-80字餐饮朋友圈文案，适合实体店老板直接复制发布，接地气有吸引力"
            )
            agent_prompt = f"用户上传菜品图片，图片分析结果：{img_analysis_result}，整理成友好客服回复交付用户"
            reply_text, updated_history, tool_log_data = text_agent_chat(agent_prompt, st.session_state.chat_history)
        # 更新对话历史
        st.session_state.chat_history = updated_history
        st.session_state.chat_history.append({"role": "user", "content": f"[菜品图片] {img_analysis_result}"})
        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
        ai_msg_item = {"role": "assistant", "content": reply_text}
        if tool_log_data:
            ai_msg_item["tool_log"] = tool_log_data
        st.session_state.ui_messages.append(ai_msg_item)
        st.session_state.last_handle_img_id = current_img_tag
        st.rerun()
