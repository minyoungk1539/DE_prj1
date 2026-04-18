import os
import streamlit as st
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict
from pymongo import MongoClient
import plotly.graph_objects as go
import numpy as np

# ── 1. 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="부작용 네트워크 분석",
    page_icon="💊",
    layout="wide",
)

st.title("💊 위고비 vs 마운자로 — 부작용 네트워크 분석")
st.caption("Reddit (10,000건 샘플링) & X (전수) 데이터 기반")

# ── 2. DB 연결 및 데이터 로드 ──────────────────────────────────
@st.cache_resource
def get_cosmos_client():
    MONGO_URI = "mongodb+srv://DEproject1:sksmsskawo123!@nje-cluster.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000&connectTimeoutMS=30000&socketTimeoutMS=60000"
    return MongoClient(MONGO_URI)

# ✅ 핵심 수정: drug_key를 캐시 키에 포함시켜 약물 전환 시 재로드 보장
@st.cache_data(ttl=3600)
def load_data_by_platform(col_name, current_drug_key):
    client = get_cosmos_client()
    db = client["DEproject"]
    collection = db[col_name]

    limit_count = 10000 if "Reddit" in col_name else 5000

    cursor = list(collection.find(
        {"side_effects": {"$ne": None, "$exists": True}, "drug_type": current_drug_key},
        {"author_id": 1, "drug_type": 1, "side_effects": 1, "_id": 1}
    ).sort("_id", -1).max_time_ms(60000).limit(limit_count))

    records = []
    for item in cursor:
        se = item.get("side_effects", "")
        effects = [e.strip().lower() for e in (se if isinstance(se, list) else str(se).split(",")) if e.strip()]
        if effects:
            records.append({
                "author_id": item.get("author_id", "unknown"),
                "drug_type": str(item.get("drug_type", "")).lower().strip(),
                "side_effects": effects,
            })
    return pd.DataFrame(records)

# ── 3. 사이드바 필터 ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    drug_choice = st.radio("약물 선택", ["위고비", "마운자로"])
    drug_key = "wegovy" if drug_choice == "위고비" else "mounjaro"

    st.divider()
    min_weight = st.slider("최소 연결 강도", 1, 150, 30)
    top_n = st.slider("상위 부작용 표시 개수 (N)", 5, 50, 12)

# ── 4. 분석 및 시각화 모듈 ──────────────────────────────────────
# ✅ 핵심 수정: 슬라이더 값(user_min_weight, user_top_n)을 인자로 받으므로
#    값이 바뀌면 캐시 없이 항상 이 함수가 재실행됩니다.
#    (st.cache_data를 이 함수에 붙이지 않는 것이 포인트)
def run_platform_analysis(df_source, platform_label, current_drug_key, user_min_weight, user_top_n):
    if df_source.empty:
        st.warning(f"{platform_label} 데이터가 없습니다.")
        return

    # ✅ 버그 수정: DB 쿼리에서 이미 drug_type 필터링했으므로 여기서 중복 필터 제거
    user_effects = (
        df_source
        .groupby("author_id")["side_effects"]
        .apply(lambda x: set(sum(x, [])))
        .to_dict()
    )

    if not user_effects:
        st.warning(f"{platform_label}: 해당 약물 데이터가 없습니다.")
        return

    # [1] 빈도 계산
    effect_freq = defaultdict(int)
    for effects in user_effects.values():
        for eff in effects:
            effect_freq[eff] += 1

    top_effects = set([e for e, _ in sorted(effect_freq.items(), key=lambda x: -x[1])[:user_top_n]])

    # [2] 엣지 계산
    edge_weights = defaultdict(int)
    for effects in user_effects.values():
        filtered = sorted(list(effects & top_effects))
        if len(filtered) > 1:
            for a, b in combinations(filtered, 2):
                edge_weights[(a, b)] += 1

    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:40]
    final_edges = {k: v for k, v in sorted_edges if v >= user_min_weight}

    # [3] 네트워크 생성
    G = nx.Graph()
    for eff in top_effects:
        G.add_node(eff, freq=effect_freq.get(eff, 0))
    for (a, b), w in final_edges.items():
        G.add_edge(a, b, weight=w)

    nodes_to_remove = [n for n, deg in G.degree() if deg <= 2]
    G.remove_nodes_from(nodes_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    # [4] 지표 출력
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("분석 대상 유저", f"{len(user_effects):,}")
    m2.metric("표시 노드", f"{len(G.nodes)}")
    m3.metric("표시 연결 수", f"{len(G.edges)}")
    m4.metric("현재 설정 강도", f"{user_min_weight}")

    if len(G.nodes) == 0:
        st.info(f"연결 강도 {user_min_weight} 이상인 데이터가 없습니다. 슬라이더를 왼쪽으로 밀어보세요.")
        return

    # [5] 시각화
    pos = nx.spring_layout(G, k=2.5, seed=42)
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    max_w, min_w = (max(weights), min(weights)) if weights else (1, 1)

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        norm_w = (d['weight'] - min_w) / (max_w - min_w + 1e-5)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=2 + (norm_w * 8), color=f'rgba(110,110,110,{0.2 + norm_w * 0.5})'),
            hoverinfo='none', mode='lines'
        ))

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=[35 + (np.log1p(G.nodes[n]['freq']) * 5) for n in G.nodes()],
            color=[G.degree(n) for n in G.nodes()],
            colorbar=dict(title="중심성", thickness=15),
            line_width=2,
            line_color='white'
        )
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            height=800,
            margin=dict(t=40, b=0, l=0, r=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='white',
            hovermode='closest'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # [6] 하단 테이블
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"🔥 {platform_label} 강한 연결 TOP 10")
        edge_data = [{"부작용 A": a, "부작용 B": b, "횟수": w} for (a, b), w in final_edges.items()]
        if edge_data:
            st.table(pd.DataFrame(edge_data).sort_values("횟수", ascending=False).head(10))
        else:
            st.info("표시할 연결 데이터가 없습니다.")
    with c2:
        st.subheader(f"🏆 {platform_label} 매개 중심성")
        between = nx.betweenness_centrality(G)
        st.dataframe(
            pd.DataFrame(between.items(), columns=["부작용", "중심성"])
            .sort_values("중심성", ascending=False)
            .head(10),
            use_container_width=True
        )

# ── 5. 메인 레이아웃 실행 ────────────────────────────────────────
t1, t2 = st.tabs(["🚀 Reddit 분석", "🐦 X 분석"])

with t1:
    data = load_data_by_platform("Reddit_Cleaned_v2", drug_key)
    run_platform_analysis(data, "Reddit", drug_key, min_weight, top_n)

with t2:
    data_x = load_data_by_platform("X_Cleaned_v2", drug_key)
    run_platform_analysis(data_x, "X", drug_key, min_weight, top_n)