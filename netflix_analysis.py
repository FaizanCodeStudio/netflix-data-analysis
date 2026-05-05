import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Synthetic Netflix Dataset ──────────────────────────────────────────────
n = 800

countries = ['United States', 'India', 'United Kingdom', 'Canada', 'France',
             'Germany', 'Japan', 'South Korea', 'Spain', 'Mexico',
             'Brazil', 'Australia', 'Italy', 'Turkey', 'Nigeria']

country_weights = [0.27, 0.14, 0.09, 0.05, 0.06, 0.05, 0.06, 0.07,
                   0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.01]

genres = ['Drama', 'Comedy', 'Action & Adventure', 'Thriller', 'Horror',
          'Romance', 'Documentary', 'Sci-Fi & Fantasy', 'Animation', 'Crime']

genre_weights = [0.22, 0.16, 0.14, 0.12, 0.08, 0.08, 0.07, 0.06, 0.05, 0.02]

content_types = ['Movie', 'TV Show']
ratings = ['G', 'PG', 'PG-13', 'TV-MA', 'TV-14', 'TV-G', 'R', 'NR']

years = np.arange(2015, 2024)
year_weights = [0.04, 0.05, 0.06, 0.09, 0.14, 0.18, 0.20, 0.14, 0.10]

data = {
    'title':        [f"Title_{i:04d}" for i in range(n)],
    'type':         np.random.choice(content_types, n, p=[0.65, 0.35]),
    'country':      np.random.choice(countries, n, p=country_weights),
    'release_year': np.random.choice(years, n, p=year_weights),
    'rating':       np.random.choice(ratings, n, p=[0.03,0.07,0.15,0.30,0.22,0.05,0.13,0.05]),
    'genre':        np.random.choice(genres, n, p=genre_weights),
    'duration_min': np.random.randint(70, 180, n),
}

month_weights = [0.06,0.06,0.08,0.07,0.07,0.09,0.10,0.08,0.09,0.10,0.11,0.09]
data['month_added'] = np.random.choice(range(1,13), n, p=month_weights)

df = pd.DataFrame(data)
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df['month_name'] = df['month_added'].map(month_names)

print("✓ Dataset created:", df.shape)
print(df.head(3).to_string())


# ── Style Config ───────────────────────────────────────────────────────────
BG       = '#0A0A0F'
SURFACE  = '#12121A'
RED      = '#E50914'
RED2     = '#FF4B4B'
GOLD     = '#F5A623'
CYAN     = '#00D4FF'
PURPLE   = '#9B59B6'
GREEN    = '#2ECC71'
TEXT     = '#FFFFFF'
SUBTEXT  = '#9999BB'

PALETTE  = [RED, CYAN, GOLD, GREEN, PURPLE, RED2,
            '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4']

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    '#2A2A3A',
    'axes.labelcolor':   SUBTEXT,
    'xtick.color':       SUBTEXT,
    'ytick.color':       SUBTEXT,
    'text.color':        TEXT,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'grid.color':        '#1E1E2E',
    'grid.linewidth':    0.6,
})

def styled_title(ax, title, size=13, color=RED):
    ax.set_title(title, fontsize=size, color=color, fontweight='bold',
                 pad=12, loc='left')

def add_value_labels(ax, bars, fmt='{:.0f}', color=TEXT, fontsize=8):
    for bar in bars:
        val = bar.get_width() if bar.get_width() else bar.get_height()
        if bar.get_width():
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    fmt.format(val), va='center', ha='left',
                    fontsize=fontsize, color=color, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    fmt.format(val), va='bottom', ha='center',
                    fontsize=fontsize, color=color, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — COUNTRY & CONTENT TYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=BG)
fig1.suptitle('NETFLIX  ·  CONTENT BY COUNTRY & TYPE',
              fontsize=18, color=TEXT, fontweight='bold', y=1.02, x=0.02, ha='left')

# 1a. Top 10 countries – horizontal bar
ax = axes[0]
top_countries = df['country'].value_counts().head(10)
colors_bar = [RED if i == 0 else PALETTE[i % len(PALETTE)] for i in range(len(top_countries))]
bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1],
               color=colors_bar[::-1], height=0.65, edgecolor='none')
for bar, color in zip(bars, colors_bar[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{int(bar.get_width())}', va='center', ha='left',
            fontsize=8, color=TEXT, fontweight='bold')
styled_title(ax, '◈ Top 10 Countries')
ax.set_xlabel('Number of Titles')
ax.axvline(top_countries.values.mean(), color=GOLD, lw=1.2, ls='--', alpha=0.7)
ax.text(top_countries.values.mean()+0.5, 0.2, 'avg', fontsize=7, color=GOLD)
ax.grid(axis='x', alpha=0.4)

# 1b. Movies vs TV Shows per country (stacked)
ax = axes[1]
top5 = df['country'].value_counts().head(7).index
ct = df[df['country'].isin(top5)].groupby(['country','type']).size().unstack(fill_value=0)
ct = ct.loc[top5]
short = [c.split()[0] for c in ct.index]
x = np.arange(len(ct))
w = 0.38
bars_m = ax.bar(x - w/2, ct.get('Movie', 0), w, label='Movie', color=RED, alpha=0.9)
bars_t = ax.bar(x + w/2, ct.get('TV Show', 0), w, label='TV Show', color=CYAN, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
styled_title(ax, '◈ Movies vs TV Shows')
ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT)
ax.grid(axis='y', alpha=0.4)

# 1c. Content rating donut
ax = axes[2]
rating_counts = df['rating'].value_counts()
wedges, texts, autotexts = ax.pie(
    rating_counts.values,
    labels=rating_counts.index,
    autopct='%1.1f%%',
    colors=PALETTE[:len(rating_counts)],
    startangle=140,
    wedgeprops={'width': 0.55, 'edgecolor': BG, 'linewidth': 1.5},
    textprops={'color': TEXT, 'fontsize': 8}
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color(BG)
    at.set_fontweight('bold')
centre = plt.Circle((0,0), 0.42, color=SURFACE)
ax.add_patch(centre)
ax.text(0, 0, 'RATING\nMIX', ha='center', va='center',
        fontsize=9, color=RED, fontweight='bold')
styled_title(ax, '◈ Content Ratings')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/netflix_country_analysis.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
print("✓ Figure 1 saved")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — GENRE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
fig2.suptitle('NETFLIX  ·  GENRE INTELLIGENCE',
              fontsize=18, color=TEXT, fontweight='bold', y=1.01, x=0.02, ha='left')

# 2a. Genre distribution bar
ax = axes[0,0]
gc = df['genre'].value_counts()
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(gc))]
bars = ax.bar(gc.index, gc.values, color=bar_colors, edgecolor='none', width=0.65)
ax.set_xticklabels(gc.index, rotation=35, ha='right', fontsize=8)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha='center', va='bottom',
            fontsize=7, color=TEXT, fontweight='bold')
styled_title(ax, '◈ Genre Distribution')
ax.set_ylabel('Titles')
ax.grid(axis='y', alpha=0.4)

# 2b. Genre heatmap: genre × year
ax = axes[0,1]
pivot = df.groupby(['genre','release_year']).size().unstack(fill_value=0)
# normalize row-wise for trend visibility
pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list('nf', [BG, RED2, GOLD], N=256)
im = ax.imshow(pivot_norm.values, aspect='auto', cmap=cmap, interpolation='nearest')
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8, label='% share within genre', 
             fraction=0.03, pad=0.03)
styled_title(ax, '◈ Genre × Year Heatmap')

# 2c. Stacked area – genre share over time
ax = axes[1,0]
top_genres = df['genre'].value_counts().head(5).index
genre_year = df[df['genre'].isin(top_genres)].groupby(['release_year','genre']).size().unstack(fill_value=0)
genre_year_pct = genre_year.div(genre_year.sum(axis=1), axis=0) * 100
area_colors = PALETTE[:len(genre_year_pct.columns)]
ax.stackplot(genre_year_pct.index, genre_year_pct.T.values,
             labels=genre_year_pct.columns, colors=area_colors, alpha=0.85)
ax.legend(loc='upper left', fontsize=7, framealpha=0, labelcolor=TEXT)
ax.set_xlabel('Release Year')
ax.set_ylabel('% Share')
styled_title(ax, '◈ Genre Share Over Time')
ax.grid(alpha=0.3)

# 2d. Genre × Type bubble chart
ax = axes[1,1]
top_g = df['genre'].value_counts().head(8).index
for i, g in enumerate(top_g):
    sub = df[df['genre'] == g]
    movies = len(sub[sub['type'] == 'Movie'])
    shows  = len(sub[sub['type'] == 'TV Show'])
    ax.scatter(movies, shows, s=(movies+shows)*2, color=PALETTE[i % len(PALETTE)],
               alpha=0.8, edgecolors=BG, linewidths=1, zorder=3)
    ax.annotate(g, (movies, shows), textcoords='offset points',
                xytext=(5,5), fontsize=7, color=TEXT)
ax.set_xlabel('Movies')
ax.set_ylabel('TV Shows')
styled_title(ax, '◈ Genre: Movies vs TV Shows')
ax.grid(alpha=0.4)
ax.plot([0, df['genre'].value_counts().max()],
        [0, df['genre'].value_counts().max()],
        '--', color=SUBTEXT, lw=0.8, alpha=0.5, label='1:1 ratio')
ax.legend(fontsize=7, framealpha=0, labelcolor=SUBTEXT)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/netflix_genre_trends.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
print("✓ Figure 2 saved")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — RELEASE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
fig3.suptitle('NETFLIX  ·  RELEASE PATTERN ANALYSIS',
              fontsize=18, color=TEXT, fontweight='bold', y=1.01, x=0.02, ha='left')

# 3a. Yearly growth
ax = axes[0,0]
yearly = df['release_year'].value_counts().sort_index()
ax.fill_between(yearly.index, yearly.values, alpha=0.3, color=RED)
ax.plot(yearly.index, yearly.values, color=RED, lw=2.5, marker='o',
        markersize=6, markerfacecolor=BG, markeredgewidth=2)
for x, y in zip(yearly.index, yearly.values):
    ax.text(x, y+1, str(y), ha='center', va='bottom', fontsize=7, color=GOLD)
styled_title(ax, '◈ Yearly Content Growth')
ax.set_xlabel('Year'); ax.set_ylabel('Titles Added')
ax.grid(alpha=0.3)

# 3b. Monthly distribution (polar-ish using bar chart with radial feel)
ax = axes[0,1]
monthly = df.groupby('month_added').size()
month_labels = [month_names[m] for m in monthly.index]
bar_c = [RED if m in [10,11,12] else CYAN if m in [6,7,8] else PALETTE[2] for m in monthly.index]
bars = ax.bar(month_labels, monthly.values, color=bar_c, edgecolor='none', width=0.7)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=7, color=TEXT)
styled_title(ax, '◈ Monthly Release Pattern')
ax.set_xlabel('Month'); ax.set_ylabel('Titles')
patches = [mpatches.Patch(color=RED, label='Q4 (Oct–Dec)'),
           mpatches.Patch(color=CYAN, label='Summer (Jun–Aug)'),
           mpatches.Patch(color=PALETTE[2], label='Other')]
ax.legend(handles=patches, fontsize=7, framealpha=0, labelcolor=TEXT)
ax.grid(axis='y', alpha=0.4)

# 3c. Movie duration histogram
ax = axes[1,0]
movies_df = df[df['type'] == 'Movie']
n_bins = 20
counts, bins, patches_h = ax.hist(movies_df['duration_min'], bins=n_bins,
                                   color=RED, alpha=0.8, edgecolor=BG, linewidth=0.5)
# gradient color
norm = plt.Normalize(counts.min(), counts.max())
for patch, cnt in zip(patches_h, counts):
    patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * norm(cnt)))
mean_dur = movies_df['duration_min'].mean()
ax.axvline(mean_dur, color=GOLD, lw=2, ls='--')
ax.text(mean_dur+1, counts.max()*0.9, f'avg {mean_dur:.0f}m',
        fontsize=8, color=GOLD, fontweight='bold')
styled_title(ax, '◈ Movie Duration Distribution')
ax.set_xlabel('Duration (minutes)'); ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.4)

# 3d. Year × Month heatmap (release cadence)
ax = axes[1,1]
cadence = df.groupby(['release_year','month_added']).size().unstack(fill_value=0)
cmap2 = mcolors.LinearSegmentedColormap.from_list('nf2', [SURFACE, RED, GOLD], N=256)
im2 = ax.imshow(cadence.values, aspect='auto', cmap=cmap2, interpolation='nearest')
ax.set_xticks(range(12))
ax.set_xticklabels([month_names[m] for m in range(1,13)], fontsize=7, rotation=45)
ax.set_yticks(range(len(cadence.index)))
ax.set_yticklabels(cadence.index, fontsize=8)
plt.colorbar(im2, ax=ax, shrink=0.8, label='Titles Released',
             fraction=0.03, pad=0.03)
styled_title(ax, '◈ Release Cadence Heatmap')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/netflix_release_patterns.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
print("✓ Figure 3 saved")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — EXECUTIVE SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(22, 14), facecolor=BG)
fig4.suptitle('NETFLIX DATA INTELLIGENCE  ·  EXECUTIVE DASHBOARD',
              fontsize=20, color=TEXT, fontweight='bold', y=1.01, x=0.01, ha='left')

gs = GridSpec(3, 4, figure=fig4, hspace=0.45, wspace=0.35)

# KPI tiles
kpis = [
    ('TOTAL TITLES', f"{n:,}", RED),
    ('COUNTRIES', f"{df['country'].nunique()}", CYAN),
    ('GENRES', f"{df['genre'].nunique()}", GOLD),
    ('YEAR SPAN', f"2015–2023", GREEN),
]
for i, (label, val, col) in enumerate(kpis):
    ax_kpi = fig4.add_subplot(gs[0, i])
    ax_kpi.set_facecolor(SURFACE)
    ax_kpi.text(0.5, 0.62, val, ha='center', va='center',
                fontsize=28, color=col, fontweight='bold', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.28, label, ha='center', va='center',
                fontsize=9, color=SUBTEXT, transform=ax_kpi.transAxes)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(1.5)

# Mini: top countries
ax5 = fig4.add_subplot(gs[1, :2])
tc = df['country'].value_counts().head(8)
bars = ax5.barh(tc.index[::-1], tc.values[::-1],
                color=[RED if i==0 else PALETTE[i%len(PALETTE)] for i in range(len(tc))][::-1],
                height=0.6, edgecolor='none')
for bar in bars:
    ax5.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             str(int(bar.get_width())), va='center', fontsize=8, color=TEXT, fontweight='bold')
styled_title(ax5, '◈ Content by Country')
ax5.grid(axis='x', alpha=0.3)

# Mini: genre pie
ax6 = fig4.add_subplot(gs[1, 2])
gc2 = df['genre'].value_counts().head(6)
wedges2, _, at2 = ax6.pie(gc2.values, labels=gc2.index, autopct='%1.0f%%',
                           colors=PALETTE[:6], startangle=90,
                           wedgeprops={'edgecolor':BG,'linewidth':1.2},
                           textprops={'fontsize':7,'color':TEXT})
for a in at2: a.set_fontsize(6); a.set_color(BG); a.set_fontweight('bold')
styled_title(ax6, '◈ Genre Mix')

# Mini: yearly trend
ax7 = fig4.add_subplot(gs[1, 3])
yt = df['release_year'].value_counts().sort_index()
ax7.fill_between(yt.index, yt.values, alpha=0.25, color=CYAN)
ax7.plot(yt.index, yt.values, color=CYAN, lw=2, marker='o',
         markersize=4, markerfacecolor=BG, markeredgewidth=1.5)
styled_title(ax7, '◈ Yearly Growth')
ax7.set_xlabel('Year', fontsize=7); ax7.tick_params(labelsize=7)
ax7.grid(alpha=0.3)

# Bottom: genre × year lines
ax8 = fig4.add_subplot(gs[2, :])
top5g = df['genre'].value_counts().head(5).index
for i, g in enumerate(top5g):
    sub = df[df['genre'] == g].groupby('release_year').size()
    ax8.plot(sub.index, sub.values, color=PALETTE[i], lw=2,
             marker='o', markersize=4, label=g, markerfacecolor=BG, markeredgewidth=1.5)
ax8.legend(fontsize=8, framealpha=0.1, labelcolor=TEXT, ncol=5,
           loc='upper left', fancybox=False)
styled_title(ax8, '◈ Top Genre Trends by Year', size=11)
ax8.set_xlabel('Release Year'); ax8.set_ylabel('Titles')
ax8.grid(alpha=0.3)

plt.savefig('/mnt/user-data/outputs/netflix_dashboard.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
print("✓ Figure 4 (Dashboard) saved")
plt.close()

print("\n══════════════════════════════════════")
print("  NETFLIX ANALYSIS COMPLETE")
print("══════════════════════════════════════")
print(f"  Total Titles    : {n}")
print(f"  Countries       : {df['country'].nunique()}")
print(f"  Genres          : {df['genre'].nunique()}")
print(f"  Top Country     : {df['country'].value_counts().idxmax()}")
print(f"  Top Genre       : {df['genre'].value_counts().idxmax()}")
print(f"  Peak Year       : {df['release_year'].value_counts().idxmax()}")
print(f"  Avg Movie Dur   : {df[df['type']=='Movie']['duration_min'].mean():.1f} min")
print("══════════════════════════════════════")
