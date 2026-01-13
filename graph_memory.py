from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

NEO_CONNECTION_URI = os.getenv("NEO_CONNECTION_URI")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")

if not NEO_CONNECTION_URI:
    raise RuntimeError("NEO_CONNECTION_URI not set")

if not NEO_USERNAME:
    raise RuntimeError("NEO_USERNAME not set")

if not NEO_PASSWORD:
    raise RuntimeError("NEO_PASSWORD not set")


driver = GraphDatabase.driver(
    NEO_CONNECTION_URI,
    auth=(NEO_USERNAME, NEO_PASSWORD)
)


def graph_add_fact(*, user_id: str, key: str, value: str):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (f:Fact {key: $key, value: $value})
            MERGE (u)-[:HAS_FACT]->(f)
            """,
            user_id=user_id,
            key=key,
            value=value
        )


def graph_add_preference(*, user_id: str, preference: str):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (p:Preference {name: $preference})
            MERGE (u)-[:HAS_PREFERENCE]->(p)
            """,
            user_id=user_id,
            preference=preference
        )


def graph_add_goal(*, user_id: str, goal: str):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (g:Goal {name: $goal})
            MERGE (u)-[:HAS_GOAL]->(g)
            """,
            user_id=user_id,
            goal=goal
        )


def graph_read_user_context(*, user_id: str):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:User {id: $user_id})-[r]->(n)
            RETURN type(r) AS relation, labels(n)[0] AS node_type, n
            """,
            user_id=user_id
        )
        return [record.data() for record in result]
