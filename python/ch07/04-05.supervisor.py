from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel


MAX_ROUNDS = 3


class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]


# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_structured_output = model.with_structured_output(SupervisorDecision)

# 사용 가능한 에이전트 정의
agents = ["researcher", "coder"]

# 시스템 프롬프트 정의
system_prompt_part_1 = (
    "당신은 다음 서브에이전트 사이의 대화를 관리하는 슈퍼바이저입니다. "
    f"서브에이전트: {agents}. 아래 사용자 요청에 따라, "
    "다음으로 행동할 서브에이전트를 지목하세요. 각 서브에이전트는 임무를 수행하고 "
    "결과와 상태를 응답합니다. 실행할 서브에이전트가 없거나 작업이 완료되면, "
    "FINISH로 응답하세요."
)

system_prompt_part_2 = (
    "위 대화를 바탕으로, 다음으로 행동할 서브에이전트는 누구입니까? "
    f"아니면 FINISH 해야 합니까? 서브에이전트: {', '.join(agents)}, FINISH"
)

supervisor_system_prompt = SystemMessage(
    content=f"{system_prompt_part_1}\n\n{system_prompt_part_2}"
)


# 에이전트 상태 정의
class AgentState(MessagesState):
    next: Literal["researcher", "coder", "FINISH"]
    rounds: int


def supervisor(state: AgentState) -> AgentState:
    # 방어 로직: 최대 라운드 도달 시 강제 종료
    if state.get("rounds", 0) >= MAX_ROUNDS:
        return {"next": "FINISH"}

    messages = [
        supervisor_system_prompt,
        *state["messages"],
    ]
    decision = model_with_structured_output.invoke(messages)
    return {"next": decision.next}


# 에이전트 함수 정의
def researcher(state: AgentState) -> AgentState:
    # 실제 구현에서는 이 함수가 리서치 작업을 수행합니다.
    # 여기서는 임의로 관련 데이터를 찾는 척 합니다.
    fake_data = "전세계 인구 데이터: [미국: 331M, 중국: 1.4B, 인도: 1.3B]"
    content = (
        "관련 데이터를 찾는 중입니다... 잠시만 기다려주세요."
        f"\n찾은 데이터: {fake_data}"
    )
    return {
        "messages": [AIMessage(content=content)],
        "rounds": state.get("rounds", 0) + 1,
        "next": "supervisor",
    }


def coder(state: AgentState) -> AgentState:
    # 실제 구현에서는 이 함수가 코드를 작성합니다.
    # 여기서는 임의로 코드를 작성하는 척 합니다.
    fake_code = """
def visualize_population(data):
    import matplotlib.pyplot as plt

    countries = list(data.keys())
    population = list(data.values())

    plt.bar(countries, population)
    plt.xlabel('Country')
    plt.ylabel('Population')
    plt.title('World Population by Country')
    plt.show()

data = {'USA': 331, 'China': 1400, 'India': 1300}
visualize_population(data)
"""
    content = "코드를 작성 중입니다... 잠시만 기다려주세요." + f"\n작성된 코드:\n{fake_code}"
    return {
        "messages": [AIMessage(content=content)],
        "rounds": state.get("rounds", 0) + 1,
        "next": "supervisor",
    }


def route_next(state: AgentState) -> Literal["researcher", "coder", END]:
    return END if state["next"] == "FINISH" else state["next"]


# 그래프 구축
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
# 슈퍼바이저의 결정에 따라 에이전트 중 하나로 라우팅하거나 종료합니다.
builder.add_conditional_edges("supervisor", route_next)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()


if __name__ == "__main__":
    initial_state: AgentState = {
        "messages": [HumanMessage(content="전세계 인구를 국적을 기준으로 시각화 해주세요.")],
        "next": "researcher",
        "rounds": 0,
    }

    for output in graph.stream(initial_state):
        node_name, node_result = next(iter(output.items()))
        print(f"\n현재 노드: {node_name}")
        if node_result.get("messages"):
            print(f"응답: {node_result['messages'][-1].content[:100]}...")
        print(f"다음 단계: {node_result.get('next', 'N/A')}")
        print(f"진행 라운드: {node_result.get('rounds', 'N/A')}/{MAX_ROUNDS}")
