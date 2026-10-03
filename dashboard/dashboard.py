"""Streamlit dashboard: Olist Brazilian E-Commerce (delivered orders), Jan 2017 - Aug 2018.

Run from the project root:  streamlit run dashboard/dashboard.py
"""
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "main_data.csv"

# Design tokens (same as the notebook)
GRAY, BLUE, RED, GREEN = "#B8BEC7", "#2F5D8C", "#C0392B", "#2E8B6B"
INK, MUTED = "#2B2F36", "#5B6270"
SOURCE = "Sumber: Olist Brazilian E-Commerce Public Dataset, pesanan delivered Jan 2017 – Agu 2018"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
REGION = {**dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
          **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
          **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
          **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
          **dict.fromkeys(["PR", "RS", "SC"], "Sul")}
DAY_LABELS = ["≤7 hari", "8–14 hari", "15–21 hari", ">21 hari"]
DIST_LABELS = ["<100 km", "100–500", "500–1.000", "1.000–2.000", ">2.000 km"]
SEG = ["Loyal (repeat)", "Baru bernilai tinggi", "Baru reguler", "Berisiko bernilai tinggi", "Hibernasi"]
SEG_LABELS = ["Loyal\n(repeat)", "Baru\nbernilai tinggi", "Baru\nreguler", "Berisiko\nbernilai tinggi", "Hibernasi"]

plt.rcParams.update({
    "figure.dpi": 100, "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8EAEE", "axes.axisbelow": True,
})
fmt_int = FuncFormatter(lambda v, _: idn(v))


def idn(x, decimals=0):
    """Format a number with Indonesian separators, e.g. 1.234,5."""
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pretty(name):
    """Readable category label."""
    return name.replace("_", " ").title()


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    """Item-level cleaned data."""
    df = pd.read_csv(DATA_PATH, parse_dates=["purchase_date"])
    df["region"] = df["customer_state"].map(REGION)
    df["month"] = df["purchase_date"].dt.to_period("M").dt.to_timestamp()
    return df


def order_table(d):
    """One row per order."""
    return (d.groupby("order_no")
             .agg(customer_no=("customer_no", "first"), purchase_date=("purchase_date", "first"), month=("month", "first"),
                  state=("customer_state", "first"), region=("region", "first"), revenue=("price", "sum"),
                  freight=("freight_value", "sum"), review=("review_score", "first"), delivery_days=("delivery_days", "first"),
                  is_late=("is_late", "first"), distance_km=("distance_km", "mean"), lat=("customer_lat", "first"),
                  lng=("customer_lng", "first"))
             .reset_index())


def state_table(o):
    """Customers, orders, revenue, delivery time and satisfaction per state."""
    s = (o.groupby("state").agg(customers=("customer_no", "nunique"), orders=("order_no", "size"), revenue=("revenue", "sum"),
                                delivery_days=("delivery_days", "mean"), review=("review", "mean"), late_pct=("is_late", "mean")))
    s["late_pct"] *= 100
    s["cust_share"] = s["customers"] / s["customers"].sum() * 100
    return s.sort_values("customers", ascending=False)


def quintile(s, labels):
    """Quintile score; falls back to rank-based bins when values tie too much."""
    try:
        return pd.qcut(s, 5, labels=labels).astype(int)
    except ValueError:
        return pd.qcut(s.rank(method="first"), 5, labels=labels).astype(int)


def rfm_table(o, snapshot):
    """RFM scores and rule-based segments per unique customer."""
    r = o.groupby("customer_no").agg(last=("purchase_date", "max"), frequency=("order_no", "nunique"), monetary=("revenue", "sum"))
    r["recency"] = (snapshot - r["last"]).dt.days
    r["R"] = quintile(r["recency"], [5, 4, 3, 2, 1])
    r["M"] = pd.qcut(r["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rules = [r["frequency"] >= 2, (r["R"] >= 4) & (r["M"] >= 4), r["R"] >= 4, r["M"] >= 4]
    r["segment"] = np.select(rules, SEG[:4], default=SEG[4])
    return r


# --------------------------------------------------------------------------- charts
def add_titles(fig, title, subtitle):
    """Left-aligned takeaway title, subtitle and source note."""
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, subtitle, ha="left", fontsize=9.5, color=MUTED)
    fig.text(0.01, 0.012, SOURCE, ha="left", fontsize=8, color=MUTED)


def finish(fig):
    fig.tight_layout(rect=[0, 0.04, 1, 0.9])
    return fig


def chart_categories(cat, total, top_n):
    """Top-N categories, Pareto curve and the five smallest categories."""
    n80 = int((cat["cum_pct"] < 80).sum()) + 1
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
    ax = axes[0]
    t = cat.head(top_n).iloc[::-1]
    colors = [GREEN if i >= len(t) - 5 else GRAY for i in range(len(t))]
    bars = ax.barh([pretty(c) for c in t.index], t["revenue"] / 1e6, color=colors, height=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlim(0, t["revenue"].max() / 1e6 * 1.18)
    ax.set_xlabel("Revenue (R$ juta)")
    ax.set_title(f"{len(t)} kategori teratas (hijau = lima teratas)", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.plot(np.arange(1, len(cat) + 1), cat["cum_pct"], color=BLUE, lw=2.6)
    ax.axhline(80, color=INK, ls="--", lw=1)
    for k, color in [(min(5, len(cat)), GREEN), (n80, RED)]:
        y = cat["cum_pct"].iloc[k - 1]
        ax.scatter([k], [y], color=color, zorder=3, s=42)
        ax.annotate(f"{k} kategori: {y:.0f}%", (k, y), xytext=(10, -16), textcoords="offset points", fontsize=9, fontweight="bold", color=color)
    ax.text(len(cat), 82, "80% revenue", ha="right", fontsize=8.5)
    ax.set_xlim(0, len(cat) + 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Jumlah kategori (urut dari revenue terbesar)")
    ax.set_ylabel("Kumulatif revenue (%)")
    ax.set_title("Konsentrasi revenue (kurva Pareto)", fontsize=10.5)

    ax = axes[2]
    b5 = cat.tail(5).iloc[::-1]
    bars = ax.barh([pretty(c) for c in b5.index], b5["revenue"] / 1e3, color=RED, height=0.65)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xlim(0, b5["revenue"].max() / 1e3 * 1.25)
    ax.set_xlabel("Revenue (R$ ribu)")
    ax.set_title("Lima kategori terendah (satuan ribu)", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    top5 = cat["share_pct"].head(5).sum()
    add_titles(fig, f"Lima kategori teratas menyumbang {top5:.0f}% revenue; {pretty(cat.index[0])} memimpin",
               f"Revenue = jumlah harga barang, total R$ {idn(total / 1e6, 2)} juta; {len(cat)} kategori (tanpa 'unknown').")
    return finish(fig)


def chart_trend(monthly):
    """Monthly orders (bars) and revenue by calendar month (lines)."""
    peak = monthly["orders"].idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(monthly))
    colors = [RED if m == peak else BLUE if m.year == 2017 else GREEN for m in monthly.index]
    ax.bar(x, monthly["orders"], color=colors, width=0.72)
    ax.set_xticks(x, [f"{MONTHS[m.month - 1]}\n{m.year}" if (m.month == 1 or i == 0) else MONTHS[m.month - 1] for i, m in enumerate(monthly.index)], fontsize=8.5)
    ax.annotate(f"Puncak: {idn(monthly.loc[peak, 'orders'])}", (list(monthly.index).index(peak), monthly.loc[peak, "orders"]),
                xytext=(-6, 6), textcoords="offset points", ha="right", color=RED, fontweight="bold")
    ax.yaxis.set_major_formatter(fmt_int)
    ax.set_ylim(0, monthly["orders"].max() * 1.15)
    ax.set_ylabel("Jumlah pesanan per bulan")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=BLUE, label="2017"), Patch(color=GREEN, label="2018"), Patch(color=RED, label="Bulan puncak")],
              frameon=False, loc="upper left")

    ax = axes[1]
    rev = monthly["revenue"] / 1e6
    for yr, color, lw in [(2017, BLUE, 2.4), (2018, GREEN, 2.8)]:
        s = rev[rev.index.year == yr]
        if len(s):
            ax.plot(s.index.month, s.values, color=color, lw=lw, marker="o", ms=4)
            ax.text(s.index.month[-1] + 0.2, s.values[-1], str(yr), color=color, va="center", fontweight="bold")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.6, 13)
    ax.set_ylim(0, rev.max() * 1.15)
    ax.set_ylabel("Revenue per bulan (R$ juta)")
    add_titles(fig, f"Pesanan memuncak pada {MONTHS[peak.month - 1]} {peak.year}",
               "Jumlah pesanan per bulan (kiri) dan revenue menurut bulan kalender 2017 vs 2018 (kanan), sesuai filter.")
    return finish(fig)


def chart_states(state, big, by_bin, min_orders):
    """Customer share, delivery time and review score by state and by delivery-time class."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.3), gridspec_kw={"width_ratios": [0.9, 1, 0.85]})
    ax = axes[0]
    t10 = state.head(10).iloc[::-1]
    bars = ax.barh(t10.index, t10["cust_share"], color=[BLUE if i >= len(t10) - 5 else GRAY for i in range(len(t10))], height=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlim(0, t10["cust_share"].max() * 1.18)
    ax.set_xlabel("Porsi pelanggan (%)")
    ax.set_title("State dengan pelanggan terbanyak", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    if len(big):
        b = big.sort_values("delivery_days")
        k = min(3, len(b) // 2)
        colors = [GREEN if i < k else RED if i >= len(b) - k else GRAY for i in range(len(b))]
        bars = ax.barh(b.index, b["delivery_days"], color=colors, height=0.72)
        ax.bar_label(bars, labels=[f"{d:.1f} hari | skor {s:.2f}" for d, s in zip(b["delivery_days"], b["review"])], padding=2, fontsize=7.5)
        ax.set_xlim(0, b["delivery_days"].max() * 1.5)
        ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("Rata-rata waktu pengiriman (hari)")
    ax.set_title(f"Waktu pengiriman dan skor ulasan (n ≥ {min_orders})", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    r = by_bin["review"]
    colors = [GREEN if v == r.max() else RED if v == r.min() else GRAY for v in r.values]
    bars = ax.bar(range(len(r)), r.values, color=colors, width=0.65)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(range(len(r)), [f"{lab}\nn={idn(c)}" for lab, c in zip(r.index, by_bin["orders"])], fontsize=8.5)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Rata-rata review score (skala 1–5)")
    ax.set_xlabel("Waktu pengiriman")
    ax.set_title("Skor ulasan menurut lama pengiriman", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    add_titles(fig, f"Pelanggan terpusat di {state.index[0]} ({state['cust_share'].iloc[0]:.0f}%); makin lama pengiriman, makin rendah skor",
               "Hijau = terbaik, merah = terburuk. State dengan pesanan sedikit tidak diperingkat pada panel tengah.")
    return finish(fig)


def chart_rfm(seg, repeat_pct, n_customers):
    """Customers vs revenue share per segment and mean spend per customer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x, w = np.arange(len(SEG)), 0.38
    ax = axes[0]
    b1 = ax.bar(x - w / 2, seg["cust_pct"], w, color=GRAY, label="% pelanggan")
    b2 = ax.bar(x + w / 2, seg["rev_pct"], w, color=GREEN, label="% revenue")
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8.5)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8.5)
    ax.set_xticks(x, SEG_LABELS, fontsize=9)
    ax.set_ylim(0, max(seg["cust_pct"].max(), seg["rev_pct"].max()) * 1.18)
    ax.set_ylabel("Persentase (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    colors = [RED if s == SEG[3] else GREEN if s == SEG[1] else GRAY for s in SEG]
    bars = ax.bar(x, seg["avg_spend"], color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"R$ {idn(v)}" for v in seg["avg_spend"]], padding=2, fontsize=9)
    ax.set_xticks(x, [f"{lab}\nn={idn(c)}" for lab, c in zip(SEG_LABELS, seg["customers"])], fontsize=8.5)
    ax.set_ylim(0, seg["avg_spend"].max() * 1.18)
    ax.set_ylabel("Rata-rata belanja per pelanggan (R$)")
    ax.grid(axis="x", visible=False)
    add_titles(fig, f"Hanya {repeat_pct:.1f}% pelanggan membeli lebih dari sekali; 'Berisiko bernilai tinggi' menyumbang {seg.loc[SEG[3], 'rev_pct']:.0f}% revenue",
               f"Segmentasi RFM {idn(n_customers)} pelanggan unik. Panel kanan: merah = prioritas win-back, hijau = peluang konversi.")
    return finish(fig)


def chart_geo(o, state, state_xy, region_tab):
    """Density of customer locations and region summary."""
    geo = o.dropna(subset=["lat", "lng"])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    hb = ax.hexbin(geo["lng"], geo["lat"], gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, shrink=0.72, label="Jumlah pesanan per sel (skala log)")
    for s in state.index[:5]:
        if s in state_xy.index:
            ax.annotate(s, (state_xy.loc[s, "lng"], state_xy.loc[s, "lat"]), fontsize=9, fontweight="bold", color=RED, ha="center",
                        va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_aspect(1 / np.cos(np.radians(-15)))
    ax.set_xlabel("Bujur")
    ax.set_ylabel("Lintang")
    ax.set_title("Kepadatan lokasi pelanggan (label = lima state teratas)", fontsize=10.5)
    ax.grid(False)

    ax = axes[1]
    colors = [GREEN if i == 0 else RED if i == len(region_tab) - 1 else GRAY for i in range(len(region_tab))]
    bars = ax.barh(region_tab.index, region_tab["delivery_days"], color=colors, height=0.62)
    ax.bar_label(bars, labels=[f"{d:.1f} hari  |  skor {s:.2f}  |  {p:.0f}% pelanggan" for d, s, p in
                               zip(region_tab["delivery_days"], region_tab["review"], region_tab["cust_share"])], padding=4, fontsize=8.5)
    ax.set_xlim(0, region_tab["delivery_days"].max() * 1.75)
    ax.set_xlabel("Rata-rata waktu pengiriman (hari)")
    ax.set_title("Wilayah: waktu kirim, skor ulasan, porsi pelanggan", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    add_titles(fig, "Kepadatan pelanggan dan ringkasan lima wilayah Brasil",
               "Kepadatan lokasi pelanggan (kiri) dan ringkasan wilayah hasil manual grouping 27 state (kanan), sesuai filter.")
    return finish(fig)


def chart_distance(by_dist):
    """Delivery time and freight share by seller-customer distance class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, col, label, fmt in [(axes[0], "days", "Rata-rata waktu pengiriman (hari)", "%.1f"),
                                (axes[1], "freight_pct", "Ongkos kirim sebagai % dari harga barang", "%.0f%%")]:
        colors = [GREEN if v == by_dist[col].min() else RED if v == by_dist[col].max() else GRAY for v in by_dist[col]]
        bars = ax.bar(range(len(by_dist)), by_dist[col], color=colors, width=0.65)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=9)
        ax.set_xticks(range(len(by_dist)), [f"{l}\n{p:.0f}% pesanan" for l, p in zip(by_dist.index, by_dist["order_pct"])], fontsize=8.5)
        ax.set_ylim(0, by_dist[col].max() * 1.18)
        ax.set_ylabel(label)
        ax.set_xlabel("Jarak penjual–pelanggan")
        ax.grid(axis="x", visible=False)
    add_titles(fig, f"Makin jauh penjual, makin lama dan mahal: {by_dist['days'].iloc[-1]:.0f} hari pada kelas terjauh vs {by_dist['days'].iloc[0]:.0f} hari pada terdekat",
               f"Jarak garis lurus berdasarkan titik tengah kode pos; n = {idn(by_dist['orders'].sum())} pesanan dengan koordinat. Hijau = terendah, merah = tertinggi.")
    return finish(fig)


def build_map(state, state_xy):
    """Interactive folium map: one bubble per state; size = orders, color = mean delivery days."""
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=state["delivery_days"].min(), vmax=state["delivery_days"].max())
    m = folium.Map(location=[-14.5, -51.0], zoom_start=4, tiles="OpenStreetMap", control_scale=True)
    for s, row in state.join(state_xy).dropna(subset=["lat"]).iterrows():
        popup = (f"<b>{s}</b> ({REGION[s]})<br>Pesanan: {idn(row['orders'])}<br>Pelanggan: {idn(row['customers'])}<br>"
                 f"Revenue: R$ {idn(row['revenue'])}<br>Waktu kirim: {row['delivery_days']:.1f} hari<br>"
                 f"Skor ulasan: {row['review']:.2f}<br>Terlambat: {row['late_pct']:.1f}%")
        folium.CircleMarker(
            location=[row["lat"], row["lng"]], radius=max(4, np.sqrt(row["orders"]) / 6), weight=1, color=INK, fill=True,
            fill_color=mcolors.to_hex(cmap(norm(row["delivery_days"]))), fill_opacity=0.8,
            tooltip=f"{s}: {idn(row['orders'])} pesanan, {row['delivery_days']:.1f} hari", popup=folium.Popup(popup, max_width=260)).add_to(m)
    return m


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def embed_html(html, height):
    """Render a raw HTML page: st.iframe on recent Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


# --------------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="🛒", layout="wide")
    df = load_data()
    d_min, d_max = df["purchase_date"].min().date(), df["purchase_date"].max().date()
    all_states = sorted(df["customer_state"].unique())
    all_cats = sorted(c for c in df["category"].unique() if c != "unknown")

    with st.sidebar:
        st.header("Filter")
        picked = st.date_input("Rentang tanggal pembelian", value=(d_min, d_max), min_value=d_min, max_value=d_max)
        states = st.multiselect("State pelanggan", all_states, default=all_states)
        cats = st.multiselect("Kategori produk (kosong = semua)", all_cats, default=[], format_func=pretty)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
        else:
            start, end = d_min, d_max
            st.info("Pilih tanggal akhir untuk menerapkan rentang; sementara seluruh periode ditampilkan.")
        st.divider()
        st.caption("**Revenue** = jumlah harga barang (tanpa ongkos kirim) pada pesanan *delivered*. **Pelanggan** dihitung dengan `customer_unique_id`.")
        st.caption("Filter kategori membatasi barang yang dihitung; pesanan dengan barang dari kategori lain hanya dihitung sebagian.")

    if not states:
        st.warning("Pilih minimal satu state pada panel filter.")
        st.stop()

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    d = df[df["customer_state"].isin(states) & (df["purchase_date"] >= t0) & (df["purchase_date"] <= t1)]
    if cats:
        d = d[d["category"].isin(cats)]
    if d.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini. Perluas rentang tanggal atau pilihan filter.")
        st.stop()
    o = order_table(d)

    st.title("🛒 Dashboard E-Commerce Olist (Brasil)")
    st.caption(f"{start:%d %b %Y} – {end:%d %b %Y} · {len(states)} state · {'semua kategori' if not cats else f'{len(cats)} kategori'} · pesanan delivered")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Revenue", f"R$ {idn(o['revenue'].sum())}")
    k2.metric("Pesanan", idn(len(o)))
    k3.metric("Pelanggan unik", idn(o["customer_no"].nunique()))
    k4.metric("Nilai rata-rata per pesanan", f"R$ {idn(o['revenue'].mean(), 1)}")
    k5.metric("Rata-rata review score", f"{o['review'].mean():.2f}", help=f"{o['is_late'].mean() * 100:.1f}% pesanan terlambat.")

    tabs = st.tabs(["1 · Kategori", "2 · Tren", "3 · Wilayah & pengiriman", "RFM", "Geospatial"])

    # ---- Q1: categories
    with tabs[0]:
        st.subheader("Kategori apa yang menghasilkan revenue terbesar dan terkecil?")
        cat = (d.groupby("category").agg(revenue=("price", "sum"), items=("price", "size")).sort_values("revenue", ascending=False))
        total_rev = d["price"].sum()
        cat = cat.drop(index="unknown", errors="ignore")
        if cat.empty:
            st.info("Tidak ada kategori yang dikenali pada filter ini.")
        else:
            cat["share_pct"] = cat["revenue"] / total_rev * 100
            cat["cum_pct"] = cat["share_pct"].cumsum()
            max_n = min(20, len(cat))
            top_n = st.slider("Jumlah kategori teratas yang ditampilkan", 5, max_n, min(10, max_n)) if max_n > 5 else max_n
            show(chart_categories(cat, total_rev, top_n))
            n80 = int((cat["cum_pct"] < 80).sum()) + 1
            st.info(f"**{pretty(cat.index[0])}** memimpin (R$ {idn(cat['revenue'].iloc[0] / 1e6, 2)} juta, {cat['share_pct'].iloc[0]:.1f}% revenue). "
                    f"Lima kategori teratas menyumbang **{cat['share_pct'].head(5).sum():.1f}%** dan **{n80} dari {len(cat)} kategori** sudah mencapai 80% revenue. "
                    f"Lima kategori terbawah hanya {cat['share_pct'].tail(5).sum():.2f}%.")

    # ---- Q2: trend
    with tabs[1]:
        st.subheader("Bagaimana tren pesanan dan revenue?")
        monthly = o.groupby("month").agg(orders=("order_no", "size"), revenue=("revenue", "sum"))
        if len(monthly) < 2:
            st.info("Pilih rentang minimal dua bulan untuk melihat tren.")
        else:
            show(chart_trend(monthly))
            peak = monthly["orders"].idxmax()
            text = f"Pesanan memuncak pada **{MONTHS[peak.month - 1]} {peak.year}** ({idn(monthly.loc[peak, 'orders'])} pesanan). "
            both = [m for m in range(1, 13) if pd.Timestamp(2017, m, 1) in monthly.index and pd.Timestamp(2018, m, 1) in monthly.index]
            if both:
                a = monthly.loc[[pd.Timestamp(2017, m, 1) for m in both]].sum()
                b = monthly.loc[[pd.Timestamp(2018, m, 1) for m in both]].sum()
                last = both[-1]
                yoy_last = (monthly.loc[pd.Timestamp(2018, last, 1), "orders"] / monthly.loc[pd.Timestamp(2017, last, 1), "orders"] - 1) * 100
                text += (f"Pada bulan yang tersedia di kedua tahun ({MONTHS[both[0] - 1]}–{MONTHS[last - 1]}), pesanan 2018 **{(b['orders'] / a['orders'] - 1) * 100:+.1f}%** "
                         f"dan revenue **{(b['revenue'] / a['revenue'] - 1) * 100:+.1f}%** dibanding 2017; pertumbuhan pada {MONTHS[last - 1]} sendiri {yoy_last:+.0f}%. "
                         "Angka YoY dipengaruhi basis awal 2017 yang kecil.")
            st.info(text)

    # ---- Q3: regions and delivery
    with tabs[2]:
        st.subheader("Di mana pelanggan terkonsentrasi dan bagaimana pengirimannya?")
        state = state_table(o)
        min_orders = 100 if len(o) >= 20000 else 20
        big = state[state["orders"] >= min_orders]
        o_bins = o.assign(bin=pd.cut(o["delivery_days"], bins=[0, 7, 14, 21, np.inf], labels=DAY_LABELS, include_lowest=True))
        by_bin = o_bins.groupby("bin", observed=True).agg(orders=("order_no", "size"), review=("review", "mean")).dropna()
        if by_bin.empty:
            st.info("Data pengiriman tidak tersedia untuk filter ini.")
        else:
            show(chart_states(state, big, by_bin, min_orders))
            late = o.groupby("is_late")["review"].mean()
            text = f"Pelanggan terbanyak di **{state.index[0]}** ({state['cust_share'].iloc[0]:.1f}%); lima state teratas mencakup {state['cust_share'].head(5).sum():.1f}%. "
            if len(big) > 1:
                text += (f"Pengiriman tercepat di **{big['delivery_days'].idxmin()}** ({big['delivery_days'].min():.1f} hari) dan terlama di "
                         f"**{big['delivery_days'].idxmax()}** ({big['delivery_days'].max():.1f} hari); skor terendah di **{big['review'].idxmin()}** ({big['review'].min():.2f}). ")
            if 0 in late.index and 1 in late.index:
                text += f"Pesanan terlambat ({o['is_late'].mean() * 100:.1f}%) dinilai {late[1]:.2f} vs {late[0]:.2f} untuk yang tepat waktu (selisih {late[0] - late[1]:.2f} poin)."
            st.info(text)

    # ---- RFM
    with tabs[3]:
        st.subheader("Segmentasi pelanggan (RFM)")
        st.caption("Acuan Recency: sehari setelah tanggal akhir filter. Recency dan Monetary diberi skor 1–5 (kuintil); pelanggan dengan ≥ 2 pesanan = loyal.")
        if o["customer_no"].nunique() < 100:
            st.info("Jumlah pelanggan terlalu sedikit untuk segmentasi RFM. Perluas filter.")
        else:
            rfm = rfm_table(o, t1 + pd.DateOffset(days=1))
            seg = (rfm.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"), avg_recency=("recency", "mean"),
                                              avg_spend=("monetary", "mean")).reindex(SEG).fillna(0))
            seg["cust_pct"] = seg["customers"] / seg["customers"].sum() * 100
            seg["rev_pct"] = seg["revenue"] / seg["revenue"].sum() * 100
            repeat_pct = (rfm["frequency"] >= 2).mean() * 100
            show(chart_rfm(seg, repeat_pct, len(rfm)))
            risk, new_hi = seg.loc[SEG[3]], seg.loc[SEG[1]]
            st.info(f"Hanya **{repeat_pct:.1f}%** pelanggan membeli lebih dari sekali. Segmen **Berisiko bernilai tinggi** ({risk['cust_pct']:.1f}% pelanggan) menyumbang "
                    f"**{risk['rev_pct']:.1f}%** revenue dan rata-rata sudah {risk['avg_recency']:.0f} hari tidak kembali (prioritas win-back); "
                    f"**Baru bernilai tinggi** ({new_hi['cust_pct']:.1f}% pelanggan, {new_hi['rev_pct']:.1f}% revenue) adalah peluang konversi menjadi pelanggan loyal.")
            st.dataframe(seg.rename(columns={"customers": "Pelanggan", "revenue": "Revenue (R$)", "avg_recency": "Rata-rata recency (hari)",
                                             "avg_spend": "Rata-rata belanja (R$)", "cust_pct": "% pelanggan", "rev_pct": "% revenue"}).round(1))
            st.download_button("Unduh tabel RFM per pelanggan (CSV)", rfm.reset_index().drop(columns=["last"]).to_csv(index=False),
                               file_name="rfm_pelanggan.csv", mime="text/csv")

    # ---- Geospatial
    with tabs[4]:
        st.subheader("Geospatial: lokasi pelanggan, wilayah, dan jarak penjual–pelanggan")
        geo = o.dropna(subset=["lat", "lng"])
        if len(geo) < 50:
            st.info("Data koordinat tidak cukup untuk peta pada filter ini.")
        else:
            state_xy = geo.groupby("state")[["lat", "lng"]].median()
            region_tab = (o.groupby("region").agg(customers=("customer_no", "nunique"), delivery_days=("delivery_days", "mean"),
                                                  review=("review", "mean")).sort_values("delivery_days"))
            region_tab["cust_share"] = region_tab["customers"] / region_tab["customers"].sum() * 100
            show(chart_geo(o, state, state_xy, region_tab))
            st.markdown("**Peta interaktif per state** (ukuran = jumlah pesanan, warna = rata-rata waktu pengiriman; klik untuk detail)")
            embed_html(build_map(state, state_xy).get_root().render(), height=520)
            o_dist = o.assign(bin=pd.cut(o["distance_km"], bins=[0, 100, 500, 1000, 2000, np.inf], labels=DIST_LABELS, include_lowest=True))
            by_dist = (o_dist.groupby("bin", observed=True).agg(orders=("order_no", "size"), days=("delivery_days", "mean"), revenue=("revenue", "sum"),
                                                                 freight=("freight", "sum")).dropna())
            if len(by_dist) >= 2:
                by_dist["freight_pct"] = by_dist["freight"] / by_dist["revenue"] * 100
                by_dist["order_pct"] = by_dist["orders"] / by_dist["orders"].sum() * 100
                show(chart_distance(by_dist))
                fast, slow = region_tab.index[0], region_tab.index[-1]
                st.info(f"Wilayah **{fast}** tercepat ({region_tab.loc[fast, 'delivery_days']:.1f} hari) dan **{slow}** terlambat ({region_tab.loc[slow, 'delivery_days']:.1f} hari). "
                        f"Pengiriman pada kelas jarak terjauh ({by_dist.index[-1]}) memakan {by_dist['days'].iloc[-1]:.1f} hari dengan ongkir "
                        f"{by_dist['freight_pct'].iloc[-1]:.0f}% dari harga barang, dibanding {by_dist['days'].iloc[0]:.1f} hari dan {by_dist['freight_pct'].iloc[0]:.0f}% pada kelas terdekat.")

    st.divider()
    st.caption(SOURCE + " · Pembersihan data dan analisis lengkap ada di notebook.ipynb.")


if __name__ == "__main__":
    main()
