from vector_memory import mem0_add, mem0_search

from graph_memory import (
    graph_add_fact,
    graph_add_preference,
    graph_add_goal,
    graph_read_user_context
)


def planner(plan: dict, user_id: str):
    intent = plan["intent"]
    memory_plan = plan["memoryPlan"]

    action = memory_plan["action"]
    vector_cfg = memory_plan["vector"]
    graph_cfg = memory_plan["graph"]
    write_cfg = memory_plan.get("write")

    result = {
        "vector": None,
        "graph": None
    }

    if intent == "memory_write" and action == "write":
        source = "explicit" if write_cfg["explicit"] else "implicit"

        if vector_cfg["use"] is True:
            mem0_add(
                user_id=user_id,
                content=write_cfg["content"],
                memory_type=write_cfg["type"],
                confidence=write_cfg["confidence"],
                source=source
            )

        if graph_cfg["use"] is True:
            try:
                result["graph"] = graph_read_user_context(
                    user_id=user_id
                )
            except Exception as e:
                result["graph"] = None


        elif write_cfg["type"] == "preference":
                graph_add_preference(
                    user_id=user_id,
                    preference=write_cfg["content"]
                )

        elif write_cfg["type"] == "goal":
                graph_add_goal(
                    user_id=user_id,
                    goal=write_cfg["content"]
                )

        return {"status": "memory_written"}

    if intent in ["query", "memory_read"] and action == "read":
        if vector_cfg["use"] is True:
            result["vector"] = mem0_search(
                user_id=user_id,
                query=vector_cfg["query"],
                limit=vector_cfg["topK"]
            )

        if graph_cfg["use"] is True:
            result["graph"] = graph_read_user_context(
                user_id=user_id
            )

        return result

    return {"status": "no_action"}
