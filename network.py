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
    page_title="부작용 네트워크 분석 리포트",
    page_icon="💊",
    layout="wide",
)

st.title("💊 위고비 & 마운자로 통합 부작용 네트워크")
st.caption("약물별 관계 분석")

# ── 2. 데이터 로드 로직 (기존과 동일) ─────────────────────────────.
@st.cache_resource
def get_cosmos_client():
    MONGO_URI = st.secrets["MONGO_URI"] 
    return MongoClient(MONGO_URI)

@st.cache_data(ttl=3600)
def load_combined_data():
    client = get_cosmos_client()
    db = client["DEproject"]
    final_records = []
    target_drugs = ["wegovy", "mounjaro"]
    target_count = 30000  # 약물별 목표 수량

    for drug in target_drugs:
        drug_data = []
        
        # 1. X_Cleaned_v2에서 해당 약물의 전체 데이터 개수 확인 (랜덤 추출용)
        # side_effects가 존재하고 drug_type이 일치하는 문서 대상
        match_condition = {
            "drug_type": drug, 
            "side_effects": {"$ne": None, "$exists": True}
        }
        
        # X에서 최대한 랜덤하게 가져오기 (최대 30,000개)
        pipeline_x = [
            {"$match": match_condition},
            {"$sample": {"size": target_count}},
            {"$project": {"author_id": 1, "side_effects": 1, "_id": 0}}
        ]
        
        results_x = list(db["X_Cleaned_v2"].aggregate(pipeline_x))
        for item in results_x:
            se = item.get("side_effects", "")
            effects = [e.strip().lower() for e in (se if isinstance(se, list) else str(se).split(",")) if e.strip()]
            if effects:
                drug_data.append({"author_id": item.get("author_id"), "drug_type": drug, "side_effects": effects})
        
        # 2. 부족분 계산 및 Reddit에서 보충
        current_count = len(drug_data)
        needed = target_count - current_count
        
        if needed > 0:
            pipeline_r = [
                {"$match": match_condition},
                {"$sample": {"size": needed}}, # 남은 수량만큼만 Reddit에서 랜덤 추출
                {"$project": {"author_id": 1, "side_effects": 1, "_id": 0}}
            ]
            results_r = list(db["Reddit_Cleaned_v2"].aggregate(pipeline_r))
            for item in results_r:
                se = item.get("side_effects", "")
                effects = [e.strip().lower() for e in (se if isinstance(se, list) else str(se).split(",")) if e.strip()]
                if effects:
                    drug_data.append({"author_id": item.get("author_id"), "drug_type": drug, "side_effects": effects})
        
        # 최종 리스트에 추가
        final_records.extend(drug_data)
        
    return pd.DataFrame(final_records)

df = load_combined_data()

# ── 3. 사이드바 필터 ────────────────────────────────────────────
with st.sidebar:
    st.header("분석 필터")
    drug_choice = st.radio("분석 대상 약물", ["전체 통합", "위고비", "마운자로"])
    st.divider()
    min_weight = st.slider("최소 연결 강도", 1, 500, 40)
    top_n = st.slider("상위 부작용 표시 개수 (N)", 5, 50, 12)

# ── 4. 데이터 처리 ─────────────────────────────────────────────
fdf = df if drug_choice == "전체 통합" else df[df["drug_type"] == ("wegovy" if drug_choice == "위고비" else "mounjaro")]
user_effects = fdf.groupby("author_id")["side_effects"].apply(lambda x: set(sum(x, []))).to_dict()

# 빈도 및 엣지 가중치 계산
effect_freq = defaultdict(int)
for effects in user_effects.values():
    for eff in effects: effect_freq[eff] += 1

top_effects = set([e for e, _ in sorted(effect_freq.items(), key=lambda x: -x[1])[:top_n]])

edge_weights = defaultdict(int)
for effects in user_effects.values():
    filtered = sorted(list(effects & top_effects))
    if len(filtered) > 1:
        for a, b in combinations(filtered, 2):
            edge_weights[(a, b)] += 1

# ── 5. 네트워크 생성 및 시각화 ────────────────────────────────────
G = nx.Graph()
for eff in top_effects: G.add_node(eff, freq=effect_freq.get(eff, 0))
for (a, b), w in edge_weights.items():
    if w >= min_weight: G.add_edge(a, b, weight=w)
G.remove_nodes_from(list(nx.isolates(G)))

if len(G.nodes) == 0:
    st.warning("데이터가 부족합니다. 연결 강도를 낮춰주세요.")
    st.stop()

pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
weights = [d['weight'] for u, v, d in G.edges(data=True)]
max_w, min_w = (max(weights), min(weights)) if weights else (1, 1)

# Plotly 시각화 로직 (눈금 제거)
edge_traces = []
for u, v, d in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    norm_w = (d['weight'] - min_w) / (max_w - min_w + 1e-5)
    edge_traces.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        line=dict(width=1 + (norm_w * 12), color=f'rgba(180, 180, 180, {0.2 + norm_w * 0.4})'),
        hoverinfo='none', mode='lines'
    ))

node_trace = go.Scatter(
    x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
    mode='markers+text', text=list(G.nodes()), textposition="top center",
    marker=dict(showscale=True, colorscale='Reds', size=[30 + (np.log1p(G.nodes[n]['freq']) * 5) for n in G.nodes()],
                color=[G.degree(n) for n in G.nodes()], colorbar=dict(title="연결수", thickness=15), line_width=2, line_color='white')
)

fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(height=700, margin=dict(t=0, b=0, l=0, r=0),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 paper_bgcolor='white', hovermode='closest'))

# ── 6. 대시보드 및 결과 테이블 ──────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("총 분석 데이터", f"{len(fdf):,}")
m2.metric("고유 유저 수", f"{len(user_effects):,}")
m3.metric("주요 부작용(Node)", len(G.nodes))
m4.metric("강력한 연결(Edge)", len(G.edges))

st.plotly_chart(fig, use_container_width=True)

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader(" 강한 연결 TOP 10")
    # ✅ 빈도수 대신 강한 연결 데이터를 가공하여 출력
    edge_list = [{"부작용 A": a, "부작용 B": b, "횟수": w} 
                 for (a, b), w in edge_weights.items()]
    top_edges_df = pd.DataFrame(edge_list).sort_values("횟수", ascending=False).head(10)
    
    if not top_edges_df.empty:
        st.table(top_edges_df)
    else:
        st.info("데이터가 없습니다.")

with c2:
    st.subheader("🏆 핵심 매개 부작용 (Network Hub)")
    
    # 1. 거리 개념 도입 (역수 취하기)
    for u, v, d in G.edges(data=True):
        d['distance'] = 1.0 / d['weight']
    
    # 2. 중심성 계산 (weight 매개변수에 distance 적용)
    between = nx.betweenness_centrality(G, weight='distance')
    
    # 3. 만약 여전히 0이라면 연결 중심성으로 자동 전환 (Fallback 로직)
    if sum(between.values()) == 0:
        between = nx.degree_centrality(G)
        st.info("💡 매개 중심성이 낮아 '연결성 중심'으로 지표를 전환했습니다.")

    centrality_df = pd.DataFrame(between.items(), columns=["부작용", "중심성"])
    # 스케일링 (가장 높은 값을 100점으로)
    max_val = centrality_df["중심성"].max()
    centrality_df["영향력 점수"] = centrality_df["중심성"].apply(lambda x: (x / max_val * 100) if max_val > 0 else 0)
    
    centrality_df = centrality_df.sort_values("영향력 점수", ascending=False).head(10)

    st.dataframe(
        centrality_df[["부작용", "영향력 점수"]], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "영향력 점수": st.column_config.ProgressColumn(
                "상대적 영향력",
                format="%.1f 점",
                min_value=0,
                max_value=100
            )
        }
    )