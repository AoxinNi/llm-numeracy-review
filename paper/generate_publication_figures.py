"""Redraw all eleven manuscript figures from unchanged evaluation snapshots.

Run with Python + matplotlib + numpy. No model calls or rescoring are performed.
Concept figures are qualitative; empirical figures preserve the original estimands.
Outputs: figures/publication/{fig*.pdf,fig*.svg,fig*.png}, manifest.json.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures' / 'publication'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT.parent / 'analysis' / 'out'
INK, MUTED, LINE = '#202B36', '#53616D', '#D7DEE4'
RG, PG = '#007A87', '#B65F22'
RGL, PGL = '#EAF5F5', '#FBF1E8'
MODELS = ['gpt', 'claude', 'gemini_minimal', 'gemini_high']
LABELS = ['GPT-5.4', 'Claude 4.6', 'Gemini MIN', 'Gemini HIGH']
COLORS = ['#2463A6', '#B65F22', '#007A87', '#7655A2']
MARKERS = ['o', 's', '^', 'D']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 600, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
manifest = {'sources': {}, 'figures': []}

def save(fig, name):
    # The Elsevier preprint text block scales these 7.1-inch assets to ~75%.
    # Keep every printed label at about 7 pt or larger after that scaling.
    for obj in fig.findobj(match=Text):
        obj.set_fontsize(max(9.5, obj.get_fontsize()))
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=.09)
    manifest['figures'].append(name)
    plt.close(fig)

def canvas(height):
    fig = plt.figure(figsize=(7.1, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    return fig, ax

def text(ax, x, y, value, size=9, color=INK, weight='normal', ha='left', va='center'):
    return ax.text(x, y, value, fontsize=max(9.5,size), color=color, weight=weight, ha=ha, va=va)

def card(ax, x, y, w, h, title, body='', color=RG, fill=RGL, size=9):
    ax.add_patch(Rectangle((x,y), w,h, facecolor=fill, edgecolor=LINE, linewidth=.7))
    ax.add_patch(Rectangle((x,y), .005,h, facecolor=color, linewidth=0))
    if body:
        text(ax, x+.018, y+h-.035, title, size, color, 'bold', va='top')
        text(ax, x+.018, y+.013, body, size, va='bottom')
    else:
        text(ax, x+w/2, y+h/2, title, size, color, 'bold', ha='center')

def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9,
                               linewidth=.85, color=color, shrinkA=2, shrinkB=2))

def grid(ax, axis='y'):
    ax.grid(axis=axis, color=LINE, linewidth=.55, alpha=.8)
    ax.set_axisbelow(True)

def accuracy_axis(ax, direction='y'):
    if direction == 'y':
        ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0,1.01,.25))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    else:
        ax.set_xlim(0, 1.03); ax.set_xticks(np.arange(0,1.01,.25))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))

def legend_models(fig, handles):
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.54, 1.005),
               ncol=4, columnspacing=1.5, handletextpad=.5)

def fig10_mitigation_matrix():
    """An open, aligned matrix with row-level color and compact method lists."""
    fig, ax = canvas(4.15)
    xs = [.18, .455, .73]
    headings = ['Existing pretrained\nmodels', 'Additional training\nor modules',
                'Training\nfrom scratch']
    statuses = ['Compatible', 'Partially compatible', 'New model required']
    for x, heading, status in zip(xs, headings, statuses):
        text(ax, x+.125, .91, heading, 10.5, INK, 'bold', ha='center')
        text(ax, x+.125, .80, status, 9.5, MUTED, ha='center')
    ax.plot([.015, .985], [.755, .755], color=INK, lw=.9)

    # Two continuous color bands replace six boxes and repeated cell headings.
    for y, h, col, fill, tag, name in [
        (.445, .285, RG, '#F0F7F7', 'RG', 'Surface form\nto value'),
        (.115, .305, PG, '#FCF5EE', 'PG', 'Arithmetic\nprocedures')]:
        ax.add_patch(Rectangle((.015, y), .97, h, facecolor=fill, edgecolor='none'))
        ax.add_patch(Rectangle((.015, y), .004, h, facecolor=col, edgecolor='none'))
        text(ax, .039, y+h*.66, tag, 13, col, 'bold')
        text(ax, .039, y+h*.40, name, 9.5, col)
        for divider in [.165, .44, .715]:
            ax.plot([divider, divider], [y+.025, y+h-.025], color=LINE, lw=.65)

    rg = [
        ['Numeracy /\nperturbation SFT', 'R2L separators'],
        ['Number Token Loss', 'Value-aware prefix / ANI', 'Place-value annotation'],
        ['Digit-level tokenization', 'xVal', 'FoNE / BitTokens']]
    pg = [
        ['Arithmetic /\nmathematical SFT', 'Process supervision', 'Head-targeted tuning'],
        ['Reversed output / LEFT', 'Integrated calculator'],
        ['Abacus embeddings', 'Position coupling', 'Randomized PE / NoPE', 'Looped transformers']]
    # Lists share their first baseline; wrapped entries get an extra line of space.
    for groups, top, color in [(rg, .678, RG), (pg, .365, PG)]:
        for x, methods in zip(xs, groups):
            y = top
            for method in methods:
                ax.plot(x+.009, y-.012, 'o', color=color, ms=2.8, markeredgewidth=0)
                text(ax, x+.026, y, method, 9.5, va='top')
                y -= .059 + .041*method.count('\n')

    ax.plot([.015, .985], [.085, .085], color=LINE, lw=.75)
    text(ax, .025, .035, 'Inference-time support', 9.5, MUTED, 'bold')
    text(ax, .37, .035, 'CoT  ·  self-consistency  ·  tools (PAL / PoT)', 9.5)
    save(fig, 'fig10_mitigation_matrix')

def concepts():
    # NGF: aligned parallel branches; no arrows through the content.
    fig, ax = canvas(4.6)
    card(ax,.23,.86,.54,.105,'Numerical Grounding Framework (NGF)',color=INK,fill='#F1F4F7',size=11)
    for x, col, fill, title, body in [
        (.02,RG,RGL,'Representational grounding (RG)','Surface form → value, magnitude, order\nand equivalent formats'),
        (.53,PG,PGL,'Procedural grounding (PG)','Faithful arithmetic procedures\nand generalization to longer operands')]:
        card(ax,x,.66,.45,.155,title,body,col,fill)
        arrow(ax,(.5,.86),(x+.225,.82),col)
    rows = [
        ('FAILURES','Tokenization and segmentation artifacts;\nformat and unit equivalence',
         'Length limits; operation asymmetry;\nerror accumulation; value dependence',.155),
        ('ROOT CAUSES','Subword segmentation; embedding geometry','Position encoding; generation order',.112),
        ('MITIGATIONS','Digit tokenization; xVal; numeracy SFT','LEFT; Abacus embeddings; process rewards',.112)]
    y=.57
    for lab,left,right,h in rows:
        y-=h
        for x,col,fill,body in [(.02,RG,RGL,left),(.53,PG,PGL,right)]:
            card(ax,x,y,.45,h,lab,body,col,fill,size=8.5)
        y-=.02
    arrow(ax,(.245,.66),(.245,.57),RG); arrow(ax,(.755,.66),(.755,.57),PG)
    text(ax,.5,.105,'Outside NGF (upstream): operand selection, i.e. using the wrong quantities',8,MUTED,ha='center')
    card(ax,.18,.0,.64,.075,'Robust numeracy requires both RG and PG',color=INK,fill='#F1F4F7')
    save(fig,'fig01_ngf_framework')

    # Pipeline: five columns with separate stage and failure rows.
    fig,ax=canvas(3.1)
    stages=[('Numeral string','9.11; 1/2\n5e-1; 50%'),('Tokenization','Frequency-based\nsubword segments'),
            ('Embeddings','Categorical\ntoken vectors'),('Position +\nattention','Column alignment\nCarry propagation'),('Decoding','Left-to-right\nanswer generation')]
    xs=np.linspace(.008,.812,5); w=.18
    for i,(x,(title,body)) in enumerate(zip(xs,stages)):
        card(ax,x,.57,w,.28,title,body,INK,'#F1F4F7',8.3)
        if i<4: arrow(ax,(x+w,.71),(xs[i+1],.71))
    failures=[('RG','Inconsistent\nsegmentation'),('RG','Unconstrained\nvalue geometry'),
              ('PG','Position shift\nbreaks alignment'),('PG','Output order vs.\narithmetic order')]
    for x,(tag,body) in zip(xs[1:],failures):
        col,fill=(RG,RGL) if tag=='RG' else (PG,PGL)
        card(ax,x,.14,w,.28,f'{tag} failure\nchannel',body,col,fill,8.3)
        arrow(ax,(x+w/2,.56),(x+w/2,.43),col)
    text(ax,.01,.94,'PROCESSING STAGES',8,MUTED,'bold')
    text(ax,.01,.045,'RG: surface form to value     |     PG: arithmetic procedure',8,MUTED)
    save(fig,'fig02_llm_number_pipeline')

    # Qualitative landscape: no numeric ticks implying measured coordinates.
    fig,ax=plt.subplots(figsize=(7.1,4.0))
    fig.subplots_adjust(left=.15,right=.97,bottom=.17,top=.93)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[.02,.98],yticks=[.08,.92])
    ax.set_xticklabels(['Atomic numerical primitives','Contextual word problems'])
    ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[1].set_ha('right')
    ax.set_yticklabels(['RG-heavy','PG-heavy'],rotation=90,va='center')
    ax.set_xlabel('Task ecology',labelpad=10); ax.set_ylabel('Dominant grounding',labelpad=10)
    ax.axhline(.5,color=LINE,lw=.6,ls='--'); ax.axvline(.5,color=LINE,lw=.6,ls='--')
    pts=[('Number Cookbook',.18,.55,RG),('NumericBench',.67,.48,RG),
         ('GSM-Symbolic',.8,.73,PG),('GSM-Ranges',.56,.87,PG),
         ('MGSM',.86,.29,RG),('BIG-Bench Hard',.55,.65,PG)]
    for lab,x,y,col in pts:
        ax.scatter(x,y,s=48,color=col,zorder=3,edgecolors='white',linewidth=.7)
        ax.annotate(lab,(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.text(.02,.03,'Qualitative placement; coordinates are not measured scores',fontsize=8,color=MUTED)
    save(fig,'fig03_benchmark_landscape')

    # Decimal cue, explicitly an illustrative segmentation, not a claimed tokenizer trace.
    fig,ax=canvas(3.6)
    for x,n,tokens in [(.05,'9.11',['9','.','11']),(.55,'9.9',['9','.','9'])]:
        card(ax,x,.78,.4,.15,n,color=INK,fill='#F1F4F7',size=15)
        for j,t in enumerate(tokens):
            card(ax,x+.065+j*.095,.56,.08,.12,t,color=RG,fill=RGL,size=12)
        arrow(ax,(x+.2,.77),(x+.2,.69))
    text(ax,.5,.46,'Illustrative subword segmentation (tokenizer-dependent)',8,MUTED,ha='center')
    card(ax,.05,.10,.40,.23,'Invalid suffix cue','11 > 9\nSuffix order ignores decimal place value',PG,PGL,9)
    card(ax,.55,.10,.40,.23,'Value-preserving comparison','9.11 < 9.90 = 9.9\nAlign decimal places before comparing',RG,RGL,9)
    arrow(ax,(.25,.42),(.25,.34),PG); arrow(ax,(.75,.42),(.75,.34),RG)
    save(fig,'fig09_tokenization_artifact')

    fig10_mitigation_matrix()

    fig,ax=canvas(4.5)
    steps=[('01','User query','Natural-language request with numbers',INK,'#F1F4F7'),
           ('02','Extract quantities','Normalize formats; preserve units and source spans',RG,RGL),
           ('03','Formulate computation','Choose the equation and check semantic intent',RG,RGL),
           ('04','Execute deterministic tool','Calculator, Python, database or verifier',PG,PGL),
           ('05','Verify result','Check equation, units and source consistency',PG,PGL),
           ('06','Final response','Explain the result and remaining uncertainty',INK,'#F1F4F7')]
    for i,(n,title,sub,col,fill) in enumerate(steps):
        y=.84-i*.16
        card(ax,.055,y,.62,.13,f'{n}  {title}',sub,col,fill,8.5)
        if i<5: arrow(ax,(.365,y),(.365,y-.025))
    text(ax,.72,.65,'RG remains essential',9,RG,'bold')
    text(ax,.72,.56,'Extract the correct values\nand formulate the\nintended computation.',8.5)
    text(ax,.72,.34,'Delegate arithmetic',9,PG,'bold')
    text(ax,.72,.24,'Deterministic execution\nrequires valid inputs\nand independent checks.',8.5)
    ax.plot([.7,.7],[.49,.81],color=RG,lw=1.1)
    ax.plot([.7,.7],[.17,.45],color=PG,lw=1.1)
    save(fig,'fig11_deployment_stack')

def empirical(nc,ctx):
    p=nc['primary']['per_model']
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),sharey=True)
    fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.82,wspace=.12)
    for ax,rk,title in zip(axes,['in','out'],['(a) Shorter operands','(b) Longer operands']):
        for j,(g,col,hatch) in enumerate([('RG',RG,''),('PG',PG,'///')]):
            vals=np.array([p[k][rk][g] for k in MODELS])
            cis=np.array([p[k][rk][g+'_ci'] for k in MODELS])
            x=np.arange(4)+(j-.5)*.38
            ax.bar(x,vals,.34,color=col,hatch=hatch,edgecolor='white',linewidth=.3,label=g,zorder=3)
            ax.errorbar(x,vals,yerr=[vals-cis[:,0],cis[:,1]-vals],fmt='none',ecolor=INK,capsize=2,lw=.85,zorder=4)
            for k,xi,v,hi in zip(MODELS,x,vals,cis[:,1]):
                label_y=hi+.025
                if j==1 and abs(hi-p[k][rk]['RG_ci'][1])<.07:
                    label_y=p[k][rk]['RG_ci'][1]+.10
                ax.text(xi,label_y,f'{100*v:.1f}',ha='center',fontsize=9.5)
        ax.set_xticks(range(4),['GPT-5.4','Claude\n4.6','Gemini\nMIN','Gemini\nHIGH'])
        ax.set_title(title,loc='left',pad=10,fontweight='bold'); accuracy_axis(ax); grid(ax)
    axes[0].set_ylabel('Exact-match accuracy')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.52,1.0))
    save(fig,'fig04_rg_pg_dissociation')

    ptd=nc['per_task_digit_exact']
    tasks=sorted(t for t in nc['groups']['PG'] if max(map(int,ptd[t]))>25)
    digits=sorted({int(d) for t in tasks for d in ptd[t]})
    fig,ax=plt.subplots(figsize=(7.1,3.4)); fig.subplots_adjust(left=.09,right=.99,bottom=.18,top=.83)
    ax.axvspan(min(digits)-.5,10.5,color='#F1F4F7',zorder=0)
    handles=[]
    for j,k in enumerate(MODELS):
        ys=[np.mean([ptd[t][str(d)][j] for t in tasks if str(d) in ptd[t]]) for d in digits]
        h,=ax.plot(digits,ys,color=COLORS[j],marker=MARKERS[j],ms=4,lw=1.65,markevery=(j,5),label=LABELS[j])
        handles.append(h)
    ax.axvline(10.5,color=MUTED,lw=.85,ls=(0,(4,3)))
    ax.text(2,1.06,'Shorter',fontsize=8,color=MUTED)
    ax.text(12,1.06,'Longer',fontsize=8,color=MUTED)
    ax.set(xlim=(min(digits)-.5,max(digits)+.5),xlabel='Operand length (digits)',ylabel='Mean exact-match accuracy')
    accuracy_axis(ax); grid(ax); legend_models(fig,handles)
    save(fig,'fig05_length_generalization_cliff')

    c=nc['primary']['contrasts']; tok=nc['tokens_primary']
    fig,(a,b)=plt.subplots(1,2,figsize=(7.1,3.6),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.79,wspace=.44)
    cats=[('RG','in'),('PG','in'),('RG','out'),('PG','out')]; xs=[0,1,2.5,3.5]
    vals=np.array([c[f'high_minus_minimal_{g}_{r}']['delta']*100 for g,r in cats])
    ci=np.array([c[f'high_minus_minimal_{g}_{r}']['ci'] for g,r in cats])*100
    for j,((g,_),x,v) in enumerate(zip(cats,xs,vals)):
        a.bar(x,v,.7,color=RG if g=='RG' else PG,hatch='' if g=='RG' else '///',edgecolor='white',lw=.3,zorder=3)
        a.errorbar(x,v,yerr=[[v-ci[j,0]],[ci[j,1]-v]],fmt='none',ecolor=INK,capsize=2,lw=.85,zorder=4)
        a.text(x,ci[j,1]+1.8,f'{v:+.1f}',ha='center',fontsize=8.5)
    a.set_xticks(xs,['RG','PG','RG','PG'])
    a.text(.5,-.19,'Shorter',transform=a.get_xaxis_transform(),ha='center',fontsize=8.5)
    a.text(3,-.19,'Longer',transform=a.get_xaxis_transform(),ha='center',fontsize=8.5)
    a.axhline(0,color=MUTED,lw=.7); a.set_ylabel('HIGH − MINIMAL (percentage points)')
    a.set_ylim(min(-3,float(ci.min())-3),float(ci.max())+8)
    a.set_title('(a) Accuracy gain',loc='left',fontweight='bold',pad=14); grid(a)
    left=np.zeros(2)
    for key,lab,col in [('input','Input','#BAC5CE'),('thinking','Thinking',COLORS[3]),('output','Output','#53616D')]:
        v=np.array([tok['gemini_minimal'][key],tok['gemini_high'][key]])
        b.barh([1,0],v,left=left,color=col,height=.48,label=lab,edgecolor='white',linewidth=.3,zorder=3)
        left+=v
    for y,v in zip([1,0],left): b.text(v+left.max()*.025,y,f'{v:,.0f}',va='center',fontsize=8.5)
    b.set(yticks=[1,0],yticklabels=['MINIMAL','HIGH'],xlabel='Mean tokens per request',xlim=(0,left.max()*1.28),ylim=(-.6,1.6))
    b.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    b.set_title('(b) Token cost',loc='left',fontweight='bold',pad=14)
    b.legend(loc='upper left',bbox_to_anchor=(0,-.32),ncol=3,fontsize=8,handlelength=1,columnspacing=1,borderaxespad=0)
    grid(b,'x'); save(fig,'fig06_reasoning_budget_tradeoff')

    wl=nc['well_learned_digit']
    tasks=[('max_Integer_Integer_Integer','Maximum · integer'),('max_Float_Float_Float','Maximum · float'),
           ('max_hard_Float_Float_Float','Maximum, hard · float'),('length_Integer_none_int','Length · integer'),
           ('length_Float_none_int','Length · float'),('count_Integer_int_int','Count digits · integer'),
           ('to_scient_Integer_none_ScientificNotation','To scientific · integer'),('sig_Integer_int_ScientificNotation','Significant figures · integer')]
    tasks=[(t,l) for t,l in tasks if t in wl['gpt']]
    mat=np.array([[wl[k][t]['well_learned_digit'] for k in MODELS] for t,_ in tasks],float)
    fig,ax=plt.subplots(figsize=(7.1,3.9)); fig.subplots_adjust(left=.31,right=.85,bottom=.16,top=.91)
    im=ax.pcolormesh(np.arange(5)-.5,np.arange(len(tasks)+1)-.5,mat,
                     cmap='Blues',vmin=0,vmax=max(55,mat.max()),shading='flat',
                     edgecolors='white',linewidth=.8,rasterized=False)
    ax.set(xlim=(-.5,3.5),ylim=(len(tasks)-.5,-.5))
    for i in range(len(tasks)):
        for j in range(4): ax.text(j,i,f'{int(mat[i,j])}' if mat[i,j] else '–',ha='center',va='center',fontsize=10,
                                  color='white' if im.norm(mat[i,j])>.58 else INK,weight='bold')
    ax.set_xticks(range(4),['GPT-5.4','Claude\n4.6','Gemini\nMIN','Gemini\nHIGH'])
    ax.set_yticks(range(len(tasks)),[l for _,l in tasks]); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,4,.999),minor=True); ax.set_yticks(np.arange(-.5,len(tasks),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5); ax.tick_params(which='minor',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb=fig.colorbar(im,cax=fig.add_axes([.885,.16,.023,.75]),ticks=[0,10,20,30,40,50])
    cb.set_label('Largest tested length (digits)\nwith ≥90% exact-match accuracy',labelpad=8,fontsize=8.5)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')  # Avoid hairline seams in PDF viewers.
    save(fig,'fig07_tokenizer_blindspots')

    nb=ctx['numericbench']; gsm=ctx['gsm_symbolic']
    cell=lambda d:(lambda k:(d[k]['acc'],d[k]['ci']))
    benches=[('Number Cookbook · shorter',lambda k:(p[k]['in']['all'],p[k]['in']['all_ci'])),
             ('Number Cookbook · longer',lambda k:(p[k]['out']['all'],p[k]['out']['all_ci'])),
             ('NumericBench · arithmetic',cell(nb['arithmetic'])),
             ('NumericBench · number list',cell(nb['num_list_100'])),
             ('GSM-Symbolic · main',cell(gsm['main'])),
             ('GSM-Symbolic · P2',cell(gsm['p2']))]
    n=len(benches)
    fig,ax=plt.subplots(figsize=(7.1,4.4)); fig.subplots_adjust(left=.32,right=.99,bottom=.13,top=.90)
    handles=[]
    for i in range(n):
        if i%2==0: ax.axhspan(i-.46,i+.46,color='#F1F4F7',zorder=0)
    for j,k in enumerate(MODELS):
        for i,(_,fn) in enumerate(benches):
            v,ci=fn(k); y=i+(j-1.5)*.20
            ax.errorbar(v,y,xerr=[[v-ci[0]],[ci[1]-v]],fmt='none',ecolor=COLORS[j],elinewidth=1.1,capsize=2,zorder=3)
            h,=ax.plot(v,y,marker=MARKERS[j],ms=5.5,mfc=COLORS[j],mec=COLORS[j],mew=1.1,ls='none',zorder=4)
            if i==0: handles.append(h)
    ax.set_yticks(range(n),[l for l,_ in benches]); ax.invert_yaxis()
    accuracy_axis(ax,'x'); ax.set_xlabel('Accuracy with 95% interval'); grid(ax,'x')
    legend_models(fig,handles)
    save(fig,'fig08_cross_benchmark_ranking')

def main():
    for name in ['nc_results.json','ctx_rerun_results.json']:
        manifest['sources'][name]=hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    nc=json.loads((DATA/'nc_results.json').read_text())
    ctx=json.loads((DATA/'ctx_rerun_results.json').read_text())
    concepts(); empirical(nc,ctx)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Redrawn {len(manifest["figures"])} figures: {OUT}')

if __name__=='__main__':
    main()
