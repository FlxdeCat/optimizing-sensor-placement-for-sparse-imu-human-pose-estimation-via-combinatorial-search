"""Noninteractive PNG plots; mesh geometry uses the same mean-shape SMPL as metrics."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import io,json,math,textwrap
import mapping
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from config import JOINT_NAMES,PARENTS,SETTINGS,joint_set,fingerprint,tolerance_label,CANDIDATE_POOL
from .storage import atomic_write,atomic_json,sha256_file
from .session import artifact,load_report,save_report
from .vendor_adapter import vendor,load_smpl_payload
from .metrics import project_so3,check_so3,rotation_errors
from .training import training_index,sequence_store
DISPLAY_ROTATION=np.asarray([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
SITE_COLORS=tuple(plt.get_cmap("tab20")(j) for j in range(20))+tuple(plt.get_cmap("Set2")(j) for j in range(4))
LABELS={"sip_dip":"SIP/DIP layout","upper_13":"13-sensor reference","greedy_a":"Greedy A (error only)"}
UNITS={"local_rotation_deg":"Local rotation error (°)","global_rotation_deg":"Global rotation error (°)","sip_deg":"SIP error (°)","mpjpe_cm":"MPJPE (cm)","mesh_cm":"Mesh error (cm)","jitter_km_s3":"Jitter (km/s³)"}


def label_name(label):
    if label.endswith("_swap"):return label_name(label.removesuffix("_swap"))+" + swap refinement"
    if label.startswith("greedy_b_tau_"):
        tau=float(label.removeprefix("greedy_b_tau_").replace("p",".").replace("m","-"))
        return f"Greedy B (MIC tie-breaker), τ={tau:g}°"
    return LABELS.get(label,label.replace("_"," "))


def save_figure(figure,path,dpi):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        buffer=io.BytesIO();figure.savefig(buffer,format="png",dpi=dpi,bbox_inches="tight",facecolor="white")
        atomic_write(path,lambda stream:stream.write(buffer.getvalue()))
    finally:plt.close(figure)
    return path


def gap_review_plot(report,thresholds):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for step in report["steps"]:
        errors=np.sort([r["local_rotation_deg"] for r in step["candidates"]]);gaps=errors-errors[0]
        axes[0].plot(np.arange(1,len(gaps)+1),gaps,marker=".",label=f"k={step['k']}")
        axes[1].plot(thresholds,[np.count_nonzero(gaps<=t) for t in thresholds],marker="o",label=f"k={step['k']}")
    axes[0].set(xlabel="Candidate rank within step",ylabel="Gap above lowest validation error (°)")
    axes[1].set(xlabel="Empirical tolerance τ (°)",ylabel="Candidates admitted by error band")
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle("BiRNN Greedy A: review candidate gaps before choosing τ; no statistical-equivalence claim")
    fig.tight_layout();return fig


def early_diagnostics(ctx,label,record):
    history=record["payload"]["details"]["history"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r["epoch"] for r in history],[r["mean_batch_mse"] for r in history],marker="o")
    axes[0].set(xlabel="Epoch",ylabel="Training matrix MSE")
    rows=record["payload"]["metrics"]["per_sequence"]
    axes[1].plot(np.arange(len(rows)),[r["mpjpe_cm"] for r in rows],marker=".")
    axes[1].set(xlabel="CMU validation sequence index",ylabel="MPJPE (cm)")
    fig.suptitle("BiRNN / "+label_name(label));fig.tight_layout()
    target=ctx.paths.figures/"baseline_diagnostics"/(label+"_"+record["key"][:12]+".png")
    save_figure(fig,target,ctx.settings.dpi);ctx.logger.info("PNG %s",target)


def joint_description(joint: int) -> str:
    if type(joint) is not int or not 0 <= joint < 24:
        raise ValueError("Joint description requires an index from 0 to 23.")
    name = JOINT_NAMES[joint]
    if name.startswith("L "):
        name = "Left " + name[2:]
    elif name.startswith("R "):
        name = "Right " + name[2:]
    return f"{joint}: {name}"

def display_points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("3D positions must be a finite [N,3] array.")
    return array @ DISPLAY_ROTATION.T

def pose_geometry(context, prediction: Any, target: Any) -> dict[str, Any]:
    p, t = project_so3(prediction), check_so3(target)
    if p.shape != (1, 24, 3, 3) or t.shape != p.shape:
        raise ValueError("One-frame pose views require [1,24,3,3] local rotations.")
    if not hasattr(context,"_plot_body"):
        context._plot_body=vendor("cpu").model(context.paths.smpl,use_pose_blendshape=False)
        context._plot_payload=load_smpl_payload(context.paths.smpl)
    body,payload=context._plot_body,context._plot_payload
    vertices = payload["J_regressor"].argmax(axis=1)
    with torch.no_grad():
        _, pj, pm = body.forward_kinematics(p.float().contiguous(), calc_mesh=True)
        _, tj, tm = body.forward_kinematics(t.float().contiguous(), calc_mesh=True)
    return {"predicted_joints": (pj[0] - pj[0, :1]).cpu().numpy(),
            "target_joints": (tj[0] - tj[0, :1]).cpu().numpy(),
            "predicted_vertices": (pm[0] - pj[0, :1]).cpu().numpy(),
            "target_vertices": (tm[0] - tj[0, :1]).cpu().numpy(),
            "faces": np.asarray(payload["f"], dtype=np.int64),
            "predicted_sites": (pm[0, vertices] - pj[0, :1]).cpu().numpy(),
            "target_sites": (tm[0, vertices] - tj[0, :1]).cpu().numpy(),
            "site_vertex_indices": vertices.tolist(), "rotation_error_deg": rotation_errors(p, t, project=False)}

def skeleton(axis, joints: Any, *, color: str, label: str | None = None, alpha: float = 1):
    points = display_points(joints)
    if points.shape != (24, 3):
        raise ValueError("Skeleton requires all 24 SMPL joint positions.")
    first = True
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        pair = points[[parent, child]]
        axis.plot(*pair.T, color=color, alpha=alpha, linewidth=1.8, label=label if first else None)
        first = False
    axis.scatter(*points.T, color=color, alpha=alpha, s=9)

def body_axis(axis, arrays: list[Any], *, compact: bool = False):
    points = np.concatenate([display_points(value) for value in arrays], axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    spans = np.maximum(upper - lower, .10)
    spans = spans * 1.15
    for dimension, setter in enumerate((axis.set_xlim, axis.set_ylim, axis.set_zlim)):
        setter(center[dimension] - spans[dimension] / 2, center[dimension] + spans[dimension] / 2)
    axis.set_box_aspect(spans, zoom=1.02)
    axis.set_proj_type("ortho")
    axis.view_init(elev=SETTINGS.camera_elevation, azim=SETTINGS.camera_azimuth)
    if compact:
        axis.set_axis_off()
    else:
        axis.set_xlabel("SMPL x (m)")
        axis.set_ylabel("−SMPL z (m)")
        axis.set_zlabel("SMPL y (m)")

def smpl_surface(axis, vertices: Any, faces: Any, *, color: str):
    points = display_points(vertices)
    triangles = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("SMPL vertices must be finite [V,3].")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu" or not len(triangles):
        raise ValueError("SMPL mesh needs nonempty integer [F,3] faces.")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("SMPL face index exceeds the vertex array.")
    # With shade=True, an explicit edgecolors="none" becomes an empty RGBA
    # array, which cannot broadcast against per-face lighting. Hide edges
    # after construction so only the nonempty face colors are shaded.
    surface = Poly3DCollection(points[triangles], facecolors=color,
                               linewidths=0, alpha=1, shade=True,
                               lightsource=LightSource(azdeg=315, altdeg=45), zorder=1)
    surface.set_edgecolor("none")
    axis.add_collection3d(surface)
    return surface

def sensor_markers(axis, sites: Any, joints: Any):
    selected = joint_set(joints)
    points = display_points(sites)[list(selected)]
    axis.scatter(*points.T, color=[SITE_COLORS[joint] for joint in selected],
                 edgecolors="white", linewidths=.7,
                 s=65, depthshade=False, zorder=4)

def sensor_key(axis, joints: Any):
    selected = joint_set(joints)
    axis.set_axis_off()
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7,
                      markerfacecolor=SITE_COLORS[joint], markeredgecolor="white",
                      label=joint_description(joint)) for joint in selected]
    return axis.legend(handles=handles, title="Selected IMU joints", loc="center left",
                       bbox_to_anchor=(0, .5), borderaxespad=0, ncol=1,
                       frameon=True, fancybox=True, framealpha=1, edgecolor="#b8bfc7",
                       borderpad=.7, labelspacing=.4, handlelength=1, handletextpad=.5,
                       fontsize=8 if len(selected) > 12 else 9, title_fontsize=9)


def pose_comparison(geometry,joints,title):
    fig=plt.figure(figsize=(14,6));grid=fig.add_gridspec(1,3,width_ratios=(3,3,1.8),wspace=.015)
    sites=list(joint_set(joints))
    for column,(prefix,name,color) in enumerate((("target","Reference","#c3c9d1"),("predicted","Prediction","#78a8cb"))):
        ax=fig.add_subplot(grid[column],projection="3d",computed_zorder=False)
        smpl_surface(ax,geometry[prefix+"_vertices"],geometry["faces"],color=color)
        body_axis(ax,[geometry["target_vertices"],geometry["predicted_vertices"]],compact=True)
        positions=geometry[prefix+"_sites"].copy()
        sensor_markers(ax,positions,sites);ax.set_title(name)
    key=fig.add_subplot(grid[2]);sensor_key(key,sites)
    fig.suptitle(textwrap.fill(title,135)+"\nMale mean-shape SMPL, pelvis aligned; sensor sites are joint-regressor vertex proxies",fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,top=.82,bottom=.03);return fig


def configuration_grid(geometry,rows,title):
    columns=min(2,len(rows));height=math.ceil(len(rows)/columns)
    fig=plt.figure(figsize=(7.5*columns,5.2*height+.7));grid=fig.add_gridspec(height,columns,wspace=.025,hspace=.06)
    for i,row in enumerate(rows):
        cell=grid[i//columns,i%columns].subgridspec(1,2,width_ratios=(3,1.8),wspace=0)
        ax=fig.add_subplot(cell[0],projection="3d",computed_zorder=False)
        smpl_surface(ax,geometry["target_vertices"],geometry["faces"],color="#c4cbd3")
        body_axis(ax,[geometry["target_vertices"]],compact=True);sensor_markers(ax,geometry["target_sites"],row["joints"])
        ax.set_title(textwrap.fill(label_name(row["label"]),40),fontsize=10,pad=0)
        sensor_key(fig.add_subplot(cell[1]),row["joints"])
    fig.suptitle(title+"\nIdentical mean-shape SMPL pose; boxed keys give joint indices and names",fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,top=1-.65/(5.2*height+.7),bottom=.02);return fig


class FigureSet:
    def __init__(self,ctx,analysis):
        self.ctx=ctx
        settings={"analysis":analysis["identity"],"matplotlib":matplotlib.__version__,"code":sha256_file(Path(__file__)),"mapping_code":sha256_file(Path(__file__).with_name("mapping_review.py")),"mapping":mapping.mapping_fingerprint(),"dpi":ctx.settings.dpi,
                  "page_size":ctx.settings.grid_page_size,"reference":ctx.settings.pose_reference_label,
                  "examples":ctx.settings.random_pose_examples,"example_configs":ctx.settings.random_pose_configurations,
                  "synthetic_examples":ctx.settings.make_synthetic_pose_examples,"camera":[ctx.settings.camera_elevation,ctx.settings.camera_azimuth],
                  "seed":ctx.settings.seed,"smpl":sha256_file(ctx.paths.smpl)}
        self.report=load_report(ctx,"visualizations",settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","files":{},"pose_examples":[]}
        self.directory=ctx.paths.figures/self.report["identity"][:16];self.directory.mkdir(parents=True,exist_ok=True)

    def add(self,name,creator):
        self.ctx.deadline.check();path=self.directory/(name+".png");old=self.report["files"].get(name)
        if old and path.is_file() and sha256_file(path)==old["sha256"]:return
        save_figure(creator(),path,self.ctx.settings.dpi)
        self.report["files"][name]={"file":path.relative_to(self.ctx.paths.runs).as_posix(),"sha256":sha256_file(path)}
        save_report(self.ctx,"visualizations",self.report);self.ctx.logger.info("PNG %s",path)


def bars(rows,metric,dataset):
    fig,ax=plt.subplots(figsize=(max(9,len(rows)*.65),5.5))
    vals=[r[metric] for r in rows];errors=[r.get(metric+"_seed_sd") for r in rows]
    ax.bar(np.arange(len(rows)),vals,color="#659bc1")
    for i,(v,e) in enumerate(zip(vals,errors)):
        if e is not None:ax.errorbar(i,v,yerr=e,color="black",capsize=3)
    ax.set_xticks(range(len(rows)),[label_name(r["label"]) for r in rows],rotation=30,ha="right")
    ax.set_ylabel(UNITS[metric]);ax.grid(axis="y",alpha=.2)
    ax.set_title(f"BiRNN / {dataset}: equal-sequence means; error bars only when ≥2 seeds")
    fig.tight_layout();return fig


def heatmap(rows,title):
    matrix=np.zeros((len(rows),24),dtype=int)
    for i,row in enumerate(rows):matrix[i,row["joints"]]=1
    matrix=matrix[:,list(CANDIDATE_POOL)]
    fig,ax=plt.subplots(figsize=(12,max(3,len(rows)*.35)))
    ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues",aspect="auto",interpolation="nearest")
    ax.set_xticks(range(len(CANDIDATE_POOL)),[joint_description(j) for j in CANDIDATE_POOL],rotation=65,ha="right",fontsize=8)
    ax.set_yticks(range(len(rows)),[label_name(r["label"]) for r in rows],fontsize=8);ax.set_title(title)
    fig.tight_layout();return fig


def summary_figures(ctx,analysis,writer):
    from .analysis import METRICS
    from .selection import random_settings,greedy_settings
    from config import tolerance_values
    final=[r for r in analysis["rows"] if r["dataset"]=="synthetic_validation" and r["status"]=="complete"]
    random=load_report(ctx,"random_search",random_settings(ctx));greedies=[load_report(ctx,"greedy_a",greedy_settings(ctx,"a"))]
    greedies += [load_report(ctx,tolerance_label(t),greedy_settings(ctx,"b",t)) for t in tolerance_values()]
    for metric in METRICS:
        pool=[r[metric] for r in final if r["family"]=="random" and r[metric] is not None]
        if not pool:continue
        def hist(metric=metric,pool=pool):
            fig,ax=plt.subplots(figsize=(10,5));ax.hist(pool,bins=min(10,len(pool)),color="#b5cfdf",edgecolor="white",label=f"{ctx.settings.random_draws} random layouts, pelvis fixed")
            for compared_index,row in enumerate(final):
                if row["family"]=="random" or row[metric] is None:continue
                p=row.get(metric+"_random_error_percentile");suffix="" if p is None else f"; p={p:.1f}"
                ax.axvline(row[metric],label=label_name(row["label"])+suffix,linewidth=1.4,color=SITE_COLORS[compared_index%24])
            ax.set(xlabel=UNITS[metric],ylabel="Number of draws",title="BiRNN / final-budget CMU validation")
            ax.legend(fontsize=8,title="p: random error percentile (lower is better)");fig.text(.5,.015,"p = random error percentile (lower is better); selection and reporting share CMU validation.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.04,1,1));return fig
        writer.add("random_final_"+metric,hist)
    def search_hist():
        fig,ax=plt.subplots(figsize=(9,4));ax.hist([r["local_rotation_deg"] for r in random["rows"]],bins=10,edgecolor="white")
        ax.set(xlabel="Frame-weighted local validation rotation error (°)",ylabel="Draw count",title=f"BiRNN / {ctx.settings.search_epochs}-epoch search budget")
        fig.tight_layout();return fig
    writer.add("random_search",search_hist)
    def curves():
        fig,ax=plt.subplots(figsize=(10,5))
        for g in greedies:
            label="greedy_a" if g["settings"]["variant"]=="a" else tolerance_label(g["tau_deg"])
            ax.plot([1]+[s["k"] for s in g["steps"]],[g["pelvis_only"]["local_rotation_deg"]]+[s["chosen_error_deg"] for s in g["steps"]],marker="o",label=label_name(label))
        ax.set(xlabel="Sensor count k, including fixed pelvis",ylabel="Local validation rotation error (°)",title="BiRNN greedy selections: search budget")
        ax.legend(title="A: minimum error; B: MIC within the error band\nτ: manually reviewed tolerance in degrees");ax.grid(alpha=.2);fig.tight_layout();return fig
    writer.add("greedy_error_vs_k",curves)
    def ties():
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for g in greedies[1:]:
            label=label_name(tolerance_label(g["tau_deg"]));x=[s["k"] for s in g["steps"]]
            axes[0].plot(x,[len(s["tie_set"]) for s in g["steps"]],marker="o",label=label)
            axes[1].plot(x,[int(s["tie_breaker_triggered"]) for s in g["steps"]],marker="o",label=label)
        axes[0].set(ylabel="Candidates in tolerance band",xlabel="Sensor count k")
        axes[1].set(ylabel="MIC tie-breaker triggered (0/1)",xlabel="Sensor count k",yticks=[0,1])
        for ax in axes:ax.legend(fontsize=8,title="τ: manual degree tolerance");ax.grid(alpha=.2)
        fig.suptitle("B = error band followed by MIC relevance − redundancy; τ is a manual degree tolerance")
        fig.tight_layout();return fig
    writer.add("tie_breaker_steps",ties)
    def tie_rates():
        fig,ax=plt.subplots(figsize=(8,4));paths=greedies[1:]
        rates=[np.mean([int(g["steps"][i]["tie_breaker_triggered"]) for g in paths]) for i in range(5)]
        ax.bar(range(2,7),rates);ax.set(xlabel="Sensor count k",ylabel="Fraction of configured τ paths triggering MIC",ylim=(0,1),title="BiRNN / descriptive tie-breaker trigger fraction")
        ax.text(.5,-.2,"B = error band then MIC; one seed, not a repeated-trial probability estimate",transform=ax.transAxes,ha="center",fontsize=8)
        fig.tight_layout();return fig
    writer.add("tie_breaker_rate",tie_rates)
    writer.add("random_sensor_heatmap",lambda:heatmap([r for r in final if r["family"]=="random"],"Random layouts / BiRNN / pelvis mandatory"))
    compared=[r for r in final if r["family"]!="random"]
    writer.add("selected_sensor_heatmap",lambda:heatmap(compared,"Compared layouts / BiRNN / 13 reviewed candidate sites"))
    for dataset in sorted({r["dataset"] for r in analysis["rows"]}):
        rows=[r for r in analysis["rows"] if r["dataset"]==dataset and r["family"]!="random" and r["status"]=="complete"]
        for metric in METRICS:
            valid=[r for r in rows if r[metric] is not None]
            if valid:writer.add(dataset+"_"+metric,lambda valid=valid,metric=metric,dataset=dataset:bars(valid,metric,dataset))
    def timing():
        fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(len(compared));labels=[label_name(r["label"]) for r in compared]
        axes[0].bar(x,[r["training_seconds"]/60 for r in compared]);axes[0].set_ylabel("Recorded optimizer time (minutes)")
        axes[1].bar(x,[r["parameters"] for r in compared]);axes[1].set_ylabel("Trainable parameters")
        for ax in axes:ax.set_xticks(x,labels,rotation=35,ha="right")
        fig.suptitle("BiRNN: zero-masked fixed-width input; duplicate layouts share a training")
        fig.tight_layout();return fig
    writer.add("training_time_and_parameters",timing)
    def losses():
        from .selection import final_results
        report,records=final_results(ctx);fig,ax=plt.subplots(figsize=(10,5))
        for row in report["rows"]:
            if row["family"]=="random" or row["seed"]!=ctx.settings.seed:continue
            history=records[row["run_key"]]["payload"]["details"]["history"]
            ax.plot([v["epoch"] for v in history],[v["mean_batch_mse"] for v in history],marker="o",label=label_name(row["label"]))
        ax.set(xlabel="Final-training epoch",ylabel="Mean training batch matrix MSE",title="BiRNN convergence diagnostics / seed1234")
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();return fig
    writer.add("final_training_loss",losses)
    def agreement():
        fig,ax=plt.subplots(figsize=(6,5));lookup={r["label"]:r for r in final}
        x=[r["local_rotation_deg"] for r in random["rows"]];y=[]
        for i in range(len(x)):
            sequences=[r for r in analysis["per_sequence"] if r["dataset"]=="synthetic_validation" and r["label"]==f"random_{i:02d}" and r["seed"]==ctx.settings.seed]
            y.append(sum(r["local_rotation_deg"]*r["frames"] for r in sequences)/sum(r["frames"] for r in sequences))
        ax.scatter(x,y)
        for i,(a,b) in enumerate(zip(x,y)):ax.annotate(str(i),(a,b),fontsize=7)
        ax.set(xlabel="Search: frame-weighted local error (°)",ylabel="Final: frame-weighted local error (°)",title="BiRNN budget agreement, seed1234; labels = draw indices")
        fig.tight_layout();return fig
    writer.add("search_final_agreement",agreement)
    def overlap():
        rows=analysis["overlap"];fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(len(rows)),[r["jaccard"] for r in rows]);ax.set_xticks(range(len(rows)),[label_name(r["configuration"]) for r in rows],rotation=20,ha="right")
        ax.set(ylabel="Jaccard overlap with Greedy A",ylim=(0,1),title="BiRNN selections; fixed pelvis contributes to every overlap")
        fig.tight_layout();return fig
    writer.add("selection_overlap",overlap)
    from .mic import matrix_report
    report=matrix_report(ctx)
    def micplot():
        with np.load(report["matrix_file"],allow_pickle=False) as data:matrix=data["matrix"].copy()
        fig,ax=plt.subplots(figsize=(11,9));image=ax.imshow(matrix,vmin=0,vmax=1,cmap="viridis")
        ax.set_xticks(range(len(CANDIDATE_POOL)),[joint_description(j) for j in CANDIDATE_POOL],rotation=65,ha="right",fontsize=7)
        ax.set_yticks(range(len(CANDIDATE_POOL)),[joint_description(j) for j in CANDIDATE_POOL],fontsize=7);fig.colorbar(image,ax=ax,label="Mean MIC over 12×12 channel pairs")
        ax.set_title("Training-only sensor dependence / minepy");fig.tight_layout();return fig
    writer.add("mic_matrix",micplot)
    def pipeline():
        fig,ax=plt.subplots(figsize=(13,7));ax.set_axis_off()
        blocks=[(.18,.8,"CMU AMASS → physical IMU cache\n24 sites stored; 13-site selectable pool\nCapture-group train/validation split"),
                (.72,.8,"Search: 3 epochs (FULL)\n30 random; Greedy A/B; pelvis fixed\nLocal validation rotation error"),
                (.72,.48,"Swap refinement of A and B\n35 one-swap neighbors per sweep\nStrictly lower error only"),
                (.18,.48,"Final training: 5 epochs (FULL)\nSIP/DIP; all-13; random; A/B; A/B + swap\nSame seeds and full training partition"),
                (.18,.16,"Frozen DIP-IMU / TotalCapture inference\n13 distinct mapped streams available\nDIP left-chest proxy disclosed"),
                (.72,.16,"Metric tables and PNG figures\nSearch versus final results separated\nBest/worst/random SMPL comparisons")]
        for x,y,text in blocks:ax.text(x,y,text,transform=ax.transAxes,ha="center",va="center",fontsize=10,bbox=dict(boxstyle="round,pad=.7",facecolor="#e8f0f6",edgecolor="#6d8ba0"))
        for a,b in (((.39,.8),(.5,.8)),((.72,.69),(.72,.59)),((.5,.48),(.39,.48)),((.18,.37),(.18,.27)),((.39,.16),(.5,.16))):ax.annotate("",xy=b,xytext=a,xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
        ax.set_title("13-site sparse-IMU experiment (FULL budgets)",fontsize=15);return fig
    writer.add("experimental_pipeline",pipeline)
    swap_figures(ctx,analysis,writer)
    mapping_review_figures(ctx,writer)


def pose_source(ctx,row):
    """Return physical local rotations without normalizing the targets."""
    if row["dataset"]=="synthetic_validation":
        source=next(r for r in training_index(ctx)["sequences"] if r["id"]==row["id"])
        return sequence_store(ctx).get(source)["pose"].copy(),source["sha256"]
    with np.load(artifact(ctx,row["prepared_file"],row["prepared_sha256"]),allow_pickle=False) as data:return data["pose"].copy(),row["prepared_sha256"]


def frame_errors(ctx,row):
    """Cache actual per-frame MPJPE and per-joint rotation errors for selection."""
    target,source_hash=pose_source(ctx,row)
    identity=fingerprint({"prediction":row["prediction_sha256"],"source":source_hash,"smpl":sha256_file(ctx.paths.smpl),"code":sha256_file(Path(__file__))})
    path=ctx.paths.runs/"pose_diagnostics"/(identity[:24]+".npz")
    if path.is_file():
        with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    from .metrics import geodesic_degrees
    saved=torch.load(artifact(ctx,row["prediction_file"],row["prediction_sha256"]),map_location="cpu",weights_only=True)
    prediction=project_so3(saved["predictions"]);target=check_so3(target)
    if prediction.shape!=target.shape:raise RuntimeError("Saved prediction/reference shapes differ; restore matching artifacts and rerun the producing evaluation.")
    body=vendor("cpu").model(ctx.paths.smpl,use_pose_blendshape=False);errors=[];joint_positions=[]
    with torch.no_grad():
        for start in range(0,len(target),ctx.settings.metric_chunk_frames):
            ctx.deadline.check();p=prediction[start:start+ctx.settings.metric_chunk_frames].float().contiguous();t=target[start:start+ctx.settings.metric_chunk_frames].float().contiguous()
            _,pj=body.forward_kinematics(p,calc_mesh=False);_,tj=body.forward_kinematics(t,calc_mesh=False)
            per_joint=torch.linalg.vector_norm((pj-pj[:,:1])-(tj-tj[:,:1]),dim=-1).numpy()*100
            joint_positions.append(per_joint);errors.append(per_joint.mean(-1))
    from .metrics import global_rotations
    result={"mpjpe_cm":np.concatenate(errors),"joint_rotation_deg":geodesic_degrees(prediction,target).numpy(),
            "position_error_cm":np.concatenate(joint_positions),"global_rotation_deg":geodesic_degrees(global_rotations(prediction),global_rotations(target)).numpy()}
    if not np.isclose(float(result["mpjpe_cm"].mean()),row["mpjpe_cm"],rtol=1e-5,atol=1e-4):raise RuntimeError("Pose-figure MPJPE disagrees with saved sequence metrics; restore matching predictions/body model and rerun analysis/evaluation.")
    atomic_write(path,lambda stream:np.savez_compressed(stream,**result));return result


def pose_figures(ctx,analysis,writer):
    datasets=sorted({r["dataset"] for r in analysis["per_sequence"]})
    if not ctx.settings.make_synthetic_pose_examples:datasets=[d for d in datasets if d!="synthetic_validation"]
    for dataset in datasets:
        rows=[r for r in analysis["per_sequence"] if r["dataset"]==dataset and r["seed"]==ctx.settings.seed]
        reference=[r for r in rows if r["label"]==ctx.settings.pose_reference_label and r["condition"]=="original"]
        if not reference:
            ctx.logger.warning("POSE %s: reference %s unavailable/non-realizable. Set pose_reference_label to an evaluated original label; no automatic substitution.",dataset,ctx.settings.pose_reference_label);continue
        by_error=sorted(reference,key=lambda r:(r["mpjpe_cm"],r["id"]))
        best,worst=by_error[0],by_error[-1]
        def ranking(by_error=by_error,dataset=dataset):
            fig,ax=plt.subplots(figsize=(12,5));ax.plot([r["mpjpe_cm"] for r in by_error],marker=".",linewidth=1)
            ax.set(xlabel="Sequence rank under reference layout",ylabel="Sequence MPJPE (cm)",title=f"{dataset}: reference-model sequence ranking ({label_name(ctx.settings.pose_reference_label)})")
            fig.tight_layout();return fig
        writer.add(dataset+"_sequence_ranking",ranking)
        choices=[("best",best,int(np.argmin(frame_errors(ctx,best)["mpjpe_cm"]))),("worst",worst,int(np.argmax(frame_errors(ctx,worst)["mpjpe_cm"])))]
        rng=np.random.default_rng(ctx.settings.seed)
        for i in range(ctx.settings.random_pose_examples):
            row=reference[int(rng.integers(len(reference)))];frame=int(rng.integers(row["frames"]))
            choices.append((f"random_{i+1}",row,frame))
        # Compare all nonrandom methods at exactly the reference-selected frame.
        labels=[r["label"] for r in analysis["rows"] if r["dataset"]==dataset and r["family"]!="random" and r["status"]=="complete"]
        labels=list(dict.fromkeys([ctx.settings.pose_reference_label,*labels,*ctx.settings.random_pose_configurations]))
        for kind,ref,frame in choices:
            for label in labels:
                candidates=[r for r in rows if r["label"]==label and r["id"]==ref["id"]]
                if not candidates:continue
                row=candidates[0];target,_=pose_source(ctx,row)
                saved=torch.load(artifact(ctx,row["prediction_file"],row["prediction_sha256"]),map_location="cpu",weights_only=True)
                prediction=saved["predictions"];diagnostics=frame_errors(ctx,row)
                identifier=f"{dataset}_{kind}_{label}";score=float(diagnostics["mpjpe_cm"][frame])
                title=f"{dataset} / {kind} / {ref['id']} / frame {frame} / BiRNN {label_name(label)} / MPJPE {score:.2f} cm"
                writer.add(identifier,lambda prediction=prediction,target=target,frame=frame,row=row,title=title:pose_comparison(pose_geometry(ctx,prediction[frame:frame+1],target[frame:frame+1]),row["joints"],title))
                manifest={"dataset":dataset,"kind":kind,"label":label,"sequence":ref["id"],"frame":frame,"frame_mpjpe_cm":score,"reference_sequence_mpjpe_cm":ref["mpjpe_cm"],
                          "frame_local_rotation_deg":float(diagnostics["joint_rotation_deg"][frame].mean()),
                          "frame_global_rotation_deg":float(diagnostics["global_rotation_deg"][frame].mean()),
                          "frame_root_orientation_deg":float(diagnostics["joint_rotation_deg"][frame,0]),
                          "frame_joint_position_errors_cm":diagnostics["position_error_cm"][frame].tolist(),
                          "criterion":"Lowest/highest reference-model sequence MPJPE, then lowest/highest reference-model frame MPJPE within that sequence; random examples sampled independently of error.",
                          "largest_local_errors":[{"joint":int(j),"joint_name":JOINT_NAMES[int(j)],"degrees":float(diagnostics["joint_rotation_deg"][frame,j])} for j in np.argsort(diagnostics["joint_rotation_deg"][frame])[-5:][::-1]]}
                writer.report["pose_examples"]=[p for p in writer.report["pose_examples"] if (p["dataset"],p["kind"],p["label"])!=(dataset,kind,label)]+[manifest]
            comparison_diagnostics(ctx,writer,rows,labels,ref,kind,frame,dataset)
            diag=frame_errors(ctx,ref)
            def diagnostic(diag=diag,ref=ref,kind=kind,dataset=dataset):
                fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={"height_ratios":(1,2)})
                axes[0].plot(np.arange(len(diag["mpjpe_cm"]))/ctx.settings.fps,diag["mpjpe_cm"]);axes[0].set(xlabel="Time after preprocessing (s)",ylabel="MPJPE (cm)")
                image=axes[1].imshow(diag["joint_rotation_deg"].T,aspect="auto",origin="lower",cmap="magma")
                axes[1].set_yticks(range(24),[joint_description(j) for j in range(24)],fontsize=7);axes[1].set_xlabel("Frame index after preprocessing");fig.colorbar(image,ax=axes[1],label="Local error (°)")
                fig.suptitle(f"{dataset} / {kind} / {ref['id']} / {label_name(ref['label'])}");fig.tight_layout();return fig
            writer.add(dataset+"_"+kind+"_error_diagnostics",diagnostic)
    # Pose-neutral layout pages include all methods and all 30 random draws.
    reference=torch.eye(3).repeat(1,24,1,1);geometry=pose_geometry(ctx,reference,reference)
    layouts=[r for r in analysis["rows"] if r["dataset"]=="synthetic_validation"]
    size=ctx.settings.grid_page_size
    for start in range(0,len(layouts),size):
        page=layouts[start:start+size];writer.add(f"configurations_3d_{start//size+1:02d}",lambda page=page:configuration_grid(geometry,page,"BiRNN sensor configurations"))


def swap_figures(ctx,analysis,writer):
    from .swaps import current_reports
    reports=current_reports(ctx)
    def traces():
        fig,ax=plt.subplots(figsize=(11,5))
        for label,report in reports:
            accepted=[r for r in report['rounds'] if r['accepted']]
            y=[report['initial']['local_rotation_deg']]+[r['best_neighbor']['local_rotation_deg'] for r in accepted]
            ax.plot(range(len(y)),y,marker='o',label=label_name(label))
        ax.set(xlabel='Accepted swaps (0 = forward greedy result)',ylabel='Frame-weighted local validation error (degrees)',
            title='BiRNN: error-only swap refinement inside the 13-site pool')
        ax.set_xticks(sorted({int(x) for line in ax.lines for x in line.get_xdata()}))
        ax.legend(fontsize=8,title='A: error only; B: error band + MIC before refinement');ax.grid(alpha=.2);fig.tight_layout();return fig
    writer.add('swap_error_traces',traces)
    for label,report in reports:
        for step in report['rounds']:
            def curve(step=step,label=label):
                values=np.sort([r['local_rotation_deg'] for r in step['candidates']])
                fig,ax=plt.subplots(figsize=(10,5));ax.plot(np.arange(1,len(values)+1),values,label='Candidate swap errors')
                ax.axhline(step['before']['local_rotation_deg'],color='black',linestyle='--',label='Current configuration')
                ax.set(xlabel='Candidate rank',ylabel='Local validation error (degrees)',title=f"{label_name(label)} — sweep {step['round']} ({'accepted' if step['accepted'] else 'stopping sweep'})")
                ax.legend();fig.tight_layout();return fig
            writer.add(f"{label}_sweep_{step['round']:02d}",curve)
    def final_comparison():
        rows=[r for r in analysis['rows'] if r['dataset']=='synthetic_validation' and r['label'].startswith('greedy_') and r['status']=='complete']
        return bars(rows,'local_rotation_deg','synthetic validation — original / swap; final budget')
    writer.add('swap_final_comparison',final_comparison)


def mapping_review_figures(ctx,writer):
    from .mapping_review import rows,write_review
    write_review(ctx);all_rows=rows()
    for page in range(2):
        block=all_rows[page*7:(page+1)*7]
        def table_plot(block=block,page=page):
            fig,ax=plt.subplots(figsize=(15,6));ax.axis('off')
            values=[[joint_description(r['smpl_index']),f"{r['dip_index']}: {r['dip_name']}\n[{r['dip_confidence']}]",
                f"{r['totalcapture_index']}: {r['totalcapture_name']}\n[{r['totalcapture_confidence']}]",
                'Approved chest proxy' if r['smpl_index']==9 else 'Approved segment mapping'] for r in block]
            table=ax.table(cellText=values,colLabels=['Synthetic SMPL site','DIP-IMU (13 of 17 streams)','TotalCapture (all 13 streams)','Mapping qualification'],
                cellLoc='left',colLoc='left',colWidths=[.23,.25,.27,.25],bbox=[0,.08,1,.8])
            table.auto_set_font_size(False);table.set_fontsize(10)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor('#cccccc')
                if r==0:cell.set_facecolor('#dce7ef');cell.set_text_props(weight='bold')
            fig.suptitle(f'Reviewed 13-site mapping — page {page+1}/2\nSegment correspondence; exact physical mounting coordinates not supplied.')
            fig.text(.04,.025,'Spine3 → DIP left chest is an approved asymmetric proxy. Names do not prove exact acceleration-site or frame equivalence.',fontsize=9)
            return fig
        writer.add(f'sensor_mapping_{page+1}',table_plot)


def comparison_diagnostics(ctx,writer,rows,labels,reference,kind,frame,dataset):
    """Same-frame/time comparisons, not independent cherry-picked method examples."""
    entries=[]
    for label in labels:
        match=next((r for r in rows if r['label']==label and r['id']==reference['id']),None)
        if match is not None:entries.append((label,match,frame_errors(ctx,match)))
    if not entries:return
    def curves():
        fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
        for label,row,errors in entries:
            seconds=np.arange(len(errors['mpjpe_cm']))/ctx.settings.fps
            for ax,key in zip(axes,('mpjpe_cm','joint_rotation_deg','global_rotation_deg')):
                values=errors[key];values=values.mean(-1) if values.ndim==2 else values
                ax.plot(seconds,values,label=label_name(label),linewidth=1)
        for ax,title in zip(axes,('MPJPE (cm)','Local rotation error (degrees)','Global rotation error (degrees)')):
            ax.set_ylabel(title);ax.axvline(frame/ctx.settings.fps,color='black',linestyle='--',label='Rendered frame');ax.grid(alpha=.15)
        axes[0].legend(fontsize=7,ncol=2);axes[-1].set_xlabel('Time after preprocessing (seconds)')
        fig.suptitle(f"{dataset} / {kind} / {reference['id']}: identical sequence across configurations")
        fig.tight_layout();return fig
    writer.add(dataset+'_'+kind+'_comparison_over_time',curves)
    def positions():
        matrix=np.stack([e['position_error_cm'][frame] for _,_,e in entries])
        fig,ax=plt.subplots(figsize=(15,max(3,len(entries)*.55)));image=ax.imshow(matrix,aspect='auto',cmap='magma')
        ax.set_xticks(range(24),[joint_description(j) for j in range(24)],rotation=65,ha='right',fontsize=8)
        ax.set_yticks(range(len(entries)),[label_name(label) for label,_,_ in entries],fontsize=8)
        ax.set_title(f'{dataset} / {kind} / frame {frame}: root-aligned per-joint position errors')
        fig.colorbar(image,ax=ax,label='Position error (cm)');fig.tight_layout();return fig
    writer.add(dataset+'_'+kind+'_joint_position_errors',positions)
    # One reference pose plus paginated multi-method predictions. Individual
    # side-by-side reference/prediction images are generated by pose_figures.
    size=ctx.settings.grid_page_size
    for start in range(0,len(entries),size):
        page=entries[start:start+size]
        def panel(page=page):
            fig=plt.figure(figsize=(15,5.3*len(page)))
            grid=fig.add_gridspec(len(page),3,width_ratios=(3,3,1.8),wspace=.02,hspace=.12)
            for i,(label,row,_) in enumerate(page):
                truth,_=pose_source(ctx,row)
                saved=torch.load(artifact(ctx,row['prediction_file'],row['prediction_sha256']),map_location='cpu',weights_only=True)
                geometry=pose_geometry(ctx,saved['predictions'][frame:frame+1],truth[frame:frame+1])
                for col,prefix,color,title in [(0,'target','#c3c9d1','Reference'),(1,'predicted','#78a8cb',label_name(label))]:
                    ax=fig.add_subplot(grid[i,col],projection='3d',computed_zorder=False)
                    smpl_surface(ax,geometry[prefix+'_vertices'],geometry['faces'],color=color)
                    body_axis(ax,[geometry['target_vertices'],geometry['predicted_vertices']],compact=True)
                    sensor_markers(ax,geometry[prefix+'_sites'],row['joints']);ax.set_title(textwrap.fill(title,55),fontsize=10)
                sensor_key(fig.add_subplot(grid[i,2]),row['joints'])
            fig.suptitle(f"{dataset} / {kind} / {reference['id']} / frame {frame}\nCommon frame selected by {label_name(ctx.settings.pose_reference_label)}; markers are synthetic-site proxies",fontsize=11)
            fig.subplots_adjust(left=.01,right=.99,top=1-.7/(5.3*len(page)),bottom=.02);return fig
        writer.add(f'{dataset}_{kind}_pose_comparison_page_{start//size+1}',panel)
