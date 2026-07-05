import streamlit as st
from openai import OpenAI
import json

st.set_page_config(page_title="智能操作助手", page_icon="🔧", layout="wide")

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

MODEL_NAME = "ep-20260705180241-s57gl"

orders_db = {
    "001": {"customer": "张三", "item": "酸菜鱼", "status": "配送中", "phone": "13900001111"},
    "002": {"customer": "李四", "item": "麻辣香锅", "status": "已签收", "phone": "13900002222"},
    "003": {"customer": "王五", "item": "蒜蓉小龙虾", "status": "待发货", "phone": "13900003333"},
}

def query_order(order_id):
    if order_id in orders_db:
        o = orders_db[order_id]
        return f"订单{order_id}：{o['customer']}，{o['item']}，状态：{o['status']}，电话：{o['phone']}"
    return f"未找到订单{order_id}"

def refund_order(order_id):
    if order_id in orders_db:
        orders_db[order_id]["status"] = "已退款"
        return f"订单{order_id}已退款成功，3个工作日内到账"
    return f"退款失败，未找到订单{order_id}"

def send_sms(phone, message):
    return f"短信已发送到{phone}，内容：{message}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单状态和详情",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "为订单办理退款",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "退款订单号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "给顾客发送短信通知",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "手机号"},
                    "message": {"type": "string", "description": "短信内容"}
                },
                "required": ["phone", "message"]
            }
        }
    }
]

def execute_tool(tool_name, args):
    if tool_name == "query_order":
        return query_order(args["order_id"])
    elif tool_name == "refund_order":
        return refund_order(args["order_id"])
    elif tool_name == "send_sms":
        return send_sms(args["phone"], args["message"])
    return "未知操作"

def run_agent(user_message, history):
    messages = [
        {"role": "system", "content": "你是店铺操作助手。帮顾客查订单、退款、发短信。语气亲切。"}
    ] + history + [
        {"role": "user", "content": user_message}
    ]
    
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
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            result = execute_tool(tool_name, args)
            tool_log = {"tool": tool_name, "args": args, "result": result}
            
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            
            final = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return final.choices[0].message.content, tool_log
        else:
            return msg.content, None
    
    except Exception as e:
        return f"❌ 操作失败：{str(e)}", None

if "agent_msgs" not in st.session_state:
    st.session_state.agent_msgs = []
if "agent_hist" not in st.session_state:
    st.session_state.agent_hist = []

st.title("🔧 店小秘AI · 智能操作")
st.markdown("Agent模式，自动查订单、退款、发短信")

with st.sidebar:
    st.header("📋 功能说明")
    st.markdown("""
    - 📦 查询订单
    - 💰 办理退款
    - 📱 发送通知
    
    **示例：**
    - "查订单001"
    - "退款002"
    - "通知13900001111外卖到了"
    """)
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.agent_msgs = []
        st.session_state.agent_hist = []
        st.rerun()

for msg in st.session_state.agent_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            with st.expander("🔧 操作详情"):
                st.json(msg["tool_log"])

if user_input := st.chat_input("输入你的需求..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("处理中..."):
        reply, tool_log = run_agent(user_input, st.session_state.agent_hist)
    
    with st.chat_message("assistant"):
        if reply.startswith("❌"):
            st.error(reply)
        else:
            st.markdown(reply)
            if tool_log:
                with st.expander("🔧 操作详情"):
                    st.json(tool_log)
    
    st.session_state.agent_msgs.append({"role":"user","content":user_input})
    st.session_state.agent_msgs.append({"role":"assistant","content":reply,"tool_log":tool_log})
    st.session_state.agent_hist.append({"role":"user","content":user_input})
    st.session_state.agent_hist.append({"role":"assistant","content":reply})
