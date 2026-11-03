"""User-run tests, including the numbered pipeline on generated data only.

No licensed data or GPU is needed. minepy and the project dependencies must be
installed. Generated geometry is a numeric test fixture, NOT a human SMPL mesh.
"""
from pathlib import Path
from dataclasses import replace
import importlib.util,json,logging,pickle,sys,tempfile,unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
import config


def fixture(root):
    data=root/"data";smpl=data/"smpl";smpl.mkdir(parents=True)
    # Each joint has one fully weighted vertex; all blend directions are zero.
    vertices=np.array([[.03*(j%3),.05*j,.02*(j%5)] for j in range(24)],np.float32)
    np.savez(smpl/"SMPL_male.npz",J_regressor=np.eye(24,dtype=np.float32),weights=np.eye(24,dtype=np.float32),
             posedirs=np.zeros((24,3,207),np.float32),shapedirs=np.zeros((24,3,10),np.float32),v_template=vertices,J=vertices,
             f=np.array([[0,j,j+1] for j in range(2,23)],np.int64),kintree_table=np.array([config.PARENTS,range(24)],np.int64))
    for subject in ("01","02","03"):
        folder=data/"amass"/"CMU"/subject;folder.mkdir(parents=True)
        pose=np.zeros((16,156),np.float32);pose[:,3]=np.linspace(0,.2,16)
        np.savez(folder/(subject+"_01_poses.npz"),poses=pose,trans=np.zeros((16,3),np.float32),betas=np.zeros(16,np.float32),mocap_framerate=np.array(60.))
    for subject in ("s_09","s_10"):
        folder=data/"dip_imu"/subject;folder.mkdir(parents=True)
        ori=np.tile(np.eye(3),(32,17,1,1));acc=np.zeros((32,17,3));gt=np.zeros((32,72))
        with (folder/"dummy.pkl").open("wb") as stream:pickle.dump({"gt":gt,"imu_acc":acc,"imu_ori":ori,"imu":np.concatenate((acc,ori.reshape(32,17,9)),axis=-1)},stream)
    folder=data/"totalcapture";folder.mkdir();refs=data/"totalcapture_smpl";refs.mkdir()
    labels=("Head","Sternum","Pelvis","LeftUpperArm","RightUpperArm","LeftForearm","RightForearm","LeftThigh","RightThigh","LeftShank","RightShank","LeftFoot","RightFoot")
    lines=["13 20"]
    for frame in range(20):
        lines.append(str(frame));lines += [label+" 1 0 0 0 0 9.8707 0" for label in labels]
    (folder/"s1_acting1_Xsens.sensors").write_text("\n".join(lines)+"\n")
    for suffix in ("ref","bone"):(folder/("s1_acting1_calib_imu_"+suffix+".txt")).write_text("13\n"+"\n".join(label+" 0 0 0 1" for label in labels)+"\n")
    with (refs/"s1_acting1.pkl").open("wb") as stream:pickle.dump({"poses":np.zeros((20,72))},stream)
    return data


def numbered(name,ctx):
    spec=importlib.util.spec_from_file_location("selftest_"+name.replace(".","_"),ROOT/"src"/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main(ctx)


class MetricTests(unittest.TestCase):
    def test_hand_computable(self):
        from shared.metrics import geodesic_degrees,rotation_errors,mpjpe_cm,mesh_error_cm,jitter_km_s3,project_so3,training_time_seconds
        eye=torch.eye(3,dtype=torch.float64);z90=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        self.assertAlmostEqual(float(geodesic_degrees(z90,eye)),90,places=7)
        self.assertAlmostEqual(float(geodesic_degrees(torch.diag(torch.tensor([1.,-1.,-1.])),eye)),180,places=7)
        pose=eye.repeat(5,24,1,1);self.assertEqual(rotation_errors(pose,pose)["sip_deg"],0)
        modified=pose.clone();modified[:,1]=z90;errors=rotation_errors(modified,pose)
        self.assertAlmostEqual(errors["local_rotation_deg"],90/24,places=7)
        self.assertAlmostEqual(errors["global_rotation_deg"],90*4/24,places=7)
        self.assertAlmostEqual(errors["sip_deg"],90/4,places=7)
        self.assertTrue(torch.allclose(project_so3(eye*2),eye))
        p=torch.zeros(5,24,3);t=p.clone();p[:,1:,0]=.01
        self.assertAlmostEqual(mpjpe_cm(p,t),23/24,places=5)
        self.assertAlmostEqual(mpjpe_cm(p+4,t+4),23/24,places=4)
        self.assertAlmostEqual(mesh_error_cm(p,t,p[:,0],t[:,0]),23/24,places=5)
        trajectory = torch.zeros(5, 24, 3, dtype=torch.float64)
        trajectory[:, 1:, 0] = (torch.arange(5, dtype=torch.float64) ** 3)[:, None]
        self.assertAlmostEqual(jitter_km_s3(trajectory,fps=1),6*23/24/1000,places=10)
        self.assertIsNone(jitter_km_s3(trajectory[:3]))
        self.assertEqual(training_time_seconds(.125),.125)
        with self.assertRaises(ValueError):training_time_seconds(-1)


class ModelTests(unittest.TestCase):
    def test_parameters_shapes_and_padding(self):
        from shared.model import build_model,parameter_count
        model=build_model("birnn",config.ModelSpec()).eval()
        self.assertEqual(parameter_count(model),212248)
        with torch.no_grad():
            x=torch.zeros(2,3,288);valid=torch.tensor([[True,True,False],[True,True,True]])
            result=model(x,valid)
            self.assertEqual(tuple(result.shape),(2,3,24,3,3))
            self.assertTrue(torch.allclose(result[:1,:2],model(x[:1,:2]),atol=1e-6))


class SelectionTests(unittest.TestCase):
    def test_reproducibility_mapping_and_ties(self):
        from shared.selection import draw_layouts,choose_candidate
        from shared.real_evaluation import realization
        import mapping
        mapping.validate_mapping();a=draw_layouts(1234,30)
        self.assertEqual(a,draw_layouts(1234,30));self.assertNotEqual(a,draw_layouts(1,30))
        self.assertTrue(all(len(set(s))==6 and s[0]==0 and set(s)<=set(config.CANDIDATE_POOL) for s in a))
        self.assertEqual(config.tolerance_values(.25),(.25,));self.assertEqual(config.tolerance_values([.25,.5,.25]),(.25,.5))
        with self.assertRaises(ValueError):config.tolerance_values(True)
        self.assertTrue(realization("dip",config.SIP_DIP)["realizable"])
        self.assertFalse(realization("dip",[0,4,5,12,18,19])["realizable"])
        self.assertTrue(realization("dip",config.CANDIDATE_POOL)["realizable"])
        self.assertTrue(realization("totalcapture",config.CANDIDATE_POOL)["realizable"])
        self.assertTrue(all(realization(dataset,sites)["realizable"] for dataset in ('dip','totalcapture') for sites in a))
        self.assertFalse(realization("totalcapture",[0,1,2,4,5,20])["realizable"])
        with self.assertRaises(ValueError):realization("dip",config.SIP_DIP,True)
        candidates=[{"added_joint":1,"local_rotation_deg":1.},{"added_joint":2,"local_rotation_deg":1.1}]
        matrix=np.zeros((13,13));matrix[2,3:]=.8
        chosen,details=choose_candidate(candidates,[0],.2,matrix)
        self.assertEqual(chosen["added_joint"],2);self.assertTrue(details["tie_breaker_triggered"])
        chosen,_=choose_candidate(candidates,[0],.05,matrix);self.assertEqual(chosen["added_joint"],1)


class SwapTests(unittest.TestCase):
    def test_neighborhood_and_strict_acceptance(self):
        from shared.swaps import neighbors,best_neighbor,improve
        original=[0,1,2,4,5,7];proposals=neighbors(original)
        self.assertEqual(len(proposals),35)
        self.assertEqual(len({tuple(s) for _,_,s in proposals}),35)
        for removed,added,sites in proposals:
            self.assertNotEqual(removed,0);self.assertIn(0,sites)
            self.assertEqual(set(original)-set(sites),{removed})
            self.assertEqual(set(sites)-set(original),{added})
            self.assertTrue(set(sites)<=set(config.CANDIDATE_POOL))
        self.assertTrue(improve(2.,1.,0.));self.assertFalse(improve(1.,1.,0.))
        a={"joints":[0,1,2,4,5,7],"local_rotation_deg":1.}
        b={"joints":[0,1,2,4,5,8],"local_rotation_deg":1.}
        self.assertIs(best_neighbor([b,a]),a)
        with self.assertRaises(ValueError):neighbors([0,1,2,4,5,20])
        with self.assertRaises(ValueError):best_neighbor([{"joints":a["joints"],"local_rotation_deg":float('nan')}])

    def test_interrupted_sweep_resume_and_certificate(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from shared import swaps,selection
        from shared.session import load_report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);logger=logging.getLogger('swap fixture')
            ctx=SimpleNamespace(paths=SimpleNamespace(runs=root),settings=SimpleNamespace(swap_max_accepted=None,swap_min_improvement_deg=0.),deadline=config.Deadline(None),logger=logger)
            calls=[];interrupt=[True]
            initial=[0,1,2,4,5,7]
            def oracle(ctx,sites):
                sites=list(sites);calls.append(tuple(sites))
                if interrupt[0] and len(calls)==5:raise config.StopRequested('test interruption')
                return {"joints":sites,"run_key":str(sites),"local_rotation_deg":float(sum(sites)),"training_seconds":0.}
            settings={"fixture":"deterministic sum-of-indices"}
            with patch.object(swaps,'source_reports',return_value=[('greedy_a',{'selected':initial})]),patch.object(swaps,'refinement_settings',return_value=settings),patch.object(selection,'score',side_effect=oracle):
                with self.assertRaises(config.StopRequested):swaps.run(ctx)
                first=load_report(ctx,'greedy_a_swap',settings)
                saved=[tuple(r['joints']) for r in first['pending']['candidates']]
                completed_calls={v:calls.count(v) for v in saved}
                interrupt[0]=False;swaps.run(ctx)
                final=load_report(ctx,'greedy_a_swap',settings)
                self.assertEqual(final['status'],'complete');self.assertTrue(final['local_optimum_certified'])
                self.assertEqual(final['current']['joints'],initial)
                self.assertEqual(len(final['rounds'][0]['candidates']),35)
                for v,n in completed_calls.items():self.assertEqual(calls.count(v),n)
                n=len(calls);swaps.run(ctx);self.assertEqual(n,len(calls))


    def test_accepted_swap_and_cap_is_not_certificate(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from shared import swaps,selection
        from shared.session import load_report
        with tempfile.TemporaryDirectory() as tmp:
            ctx=SimpleNamespace(paths=SimpleNamespace(runs=Path(tmp)),settings=SimpleNamespace(swap_max_accepted=1,swap_min_improvement_deg=0.),deadline=config.Deadline(None),logger=logging.getLogger('swap cap fixture'))
            initial=[0,1,2,4,5,7]
            def oracle(ctx,sites):
                return {"joints":list(sites),"run_key":str(sites),"local_rotation_deg":100.-sum(sites),"training_seconds":0.}
            settings={"fixture":"one accepted swap cap"}
            with patch.object(swaps,'source_reports',return_value=[('greedy_a',{'selected':initial})]),patch.object(swaps,'refinement_settings',return_value=settings),patch.object(selection,'score',side_effect=oracle):
                swaps.run(ctx)
            report=load_report(ctx,'greedy_a_swap',settings)
            self.assertEqual(report['accepted_swaps'],1)
            self.assertFalse(report['local_optimum_certified'])
            self.assertEqual(report['stop_reason'],'accepted_swap_cap')
            self.assertLess(report['current']['local_rotation_deg'],report['initial']['local_rotation_deg'])


class StorageTests(unittest.TestCase):
    def test_atomic_ledger_resume(self):
        from shared.storage import atomic_json,atomic_write,Ledger,RunKey
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/"numpy.json",{"flag":np.bool_(True),"n":np.int64(2)})
            self.assertIs(json.loads((root/"numpy.json").read_text())["flag"],True)
            ledger=Ledger(root/"ledger.jsonl");key=RunKey.create("birnn",[0,4],1234,{"epochs":1},data_fingerprint="dummy",pipeline_fingerprint="dummy")
            checkpoint=root/"checkpoint.pt";atomic_write(checkpoint,lambda stream:torch.save({"dummy":True},stream))
            ledger.record_completed(key,metrics={"local_rotation_deg":1.},checkpoint=checkpoint)
            self.assertIsNotNone(Ledger(ledger.path).completed(key))
            with ledger.path.open("ab") as stream:stream.write(b'{"torn":')
            self.assertEqual(len(Ledger(ledger.path).records),1)


class PipelineTests(unittest.TestCase):
    def test_generated_pipeline_and_mmap_cleanup(self):
        from shared.session import make_session
        from shared.training import training_index,sequence_store,close_sequence_store,CachedTrainingDataset
        from shared.real_data import load_tc_sequence
        from shared.data import AMASSWindowDataset
        from shared.selection import catalog
        from shared.storage import sha256_file
        ctx=make_session("selftest")
        try:
            numbered("0_check.py",ctx);numbered("1_synthesize.py",ctx);numbered("benchmark.py",ctx)
            index=training_index(ctx);budget=ctx.settings.budget("search")
            from shared.metrics import PoseMetrics
            identity=torch.eye(3).repeat(5,24,1,1)
            zero=PoseMetrics(ctx.paths.smpl,chunk_frames=2)(identity,identity)
            self.assertEqual(zero["mpjpe_cm"],0);self.assertEqual(zero["mesh_cm"],0);self.assertEqual(zero["jitter_km_s3"],0)
            from shared.training import evaluate
            class InterruptAfterInitialCheck:
                checks=0
                def check(self):
                    self.checks+=1
                    if self.checks>=2:raise config.StopRequested("generated interruption")
            ctx.deadline=InterruptAfterInitialCheck()
            with self.assertRaises(config.StopRequested):evaluate(ctx,"birnn",config.SIP_DIP,1234,stage="search")
            self.assertTrue(any(r["status"]=="interrupted" for r in ctx.ledger.records))
            ctx.deadline=config.Deadline(None)
            resumed=evaluate(ctx,"birnn",config.SIP_DIP,1234,stage="search")
            self.assertEqual(resumed["status"],"completed")
            cached=CachedTrainingDataset(ctx,index,config.SIP_DIP,budget,partition="train",split_seed=1234)
            physical=AMASSWindowDataset(ctx.paths,index,config.SIP_DIP,budget,partition="train",split_seed=1234)
            self.assertTrue(torch.equal(cached[0]["x"],physical[0]["x"]))
            close_sequence_store(ctx)
            store=sequence_store(ctx);store.cache.limit=40000
            borrowed=store.get(index["sequences"][0]);owner=borrowed["acc"]
            store.get(index["sequences"][1]);self.assertGreaterEqual(store.cache.evictions,1)
            # Explicit close releases Windows file handles even with borrowed roots.
            close_sequence_store(ctx);self.assertTrue(owner._mmap.closed)
            del borrowed,owner,cached,physical
            for name in ("2_upperbound.py","3_baseline.py","4_randoms.py","5_greedy_a.py","6_greedy_b.py","7_swap.py","8_retrain.py","9_evaluation.py","10_analyze.py","11_visualizations.py"):
                numbered(name,ctx)
            completed=[r for r in ctx.ledger.records if r["status"]=="completed"]
            weights={r["key"]:sha256_file(ctx.paths.runs/r["payload"]["checkpoint"]) for r in completed}
            numbered("8_retrain.py",ctx)
            self.assertEqual(weights,{r["key"]:sha256_file(ctx.paths.runs/r["payload"]["checkpoint"]) for r in ctx.ledger.records if r["status"]=="completed"})
            items,_=catalog(ctx);self.assertEqual(len(items),36)
            a=json.loads((ctx.paths.runs/"greedy_a.json").read_text())
            self.assertEqual([len(s["candidates"]) for s in a["steps"]],[12,11,10,9,8])
            mic=json.loads((ctx.paths.runs/"mic.json").read_text())
            with np.load(mic["matrix_file"]) as archive:matrix=archive["matrix"].copy()
            self.assertEqual(matrix.shape,(13,13));self.assertTrue(np.isfinite(matrix).all());self.assertTrue(np.allclose(matrix,matrix.T))
            self.assertEqual(len(list(ctx.paths.figures.rglob("*.png")))>20,True)
            tc=load_tc_sequence(ctx.paths.totalcapture/"s1_acting1_Xsens.sensors",ctx.paths.totalcapture_smpl/"s1_acting1.pkl",None,"s1","acting1")
            self.assertEqual(len(tc.pose),19);self.assertIsNone(tc.vicon)
        finally:
            close_sequence_store(ctx)
            for handler in list(ctx.logger.handlers):handler.close();ctx.logger.removeHandler(handler)


def main():
    torch.set_num_threads(1)  # tiny CPU fixtures; avoid thread-pool overhead
    with tempfile.TemporaryDirectory(prefix="imuplace selftest ") as tmp:
        root=Path(tmp);data=fixture(root)
        config.ACTIVE_PRESET="pilot";config.DATA_ROOT=data;config.CACHE_ROOT=root/"cache";config.RUN_ROOT=root/"run"
        config.AMASS_DIRECTORY=data/"amass"/"CMU";config.SMPL_FILE=data/"smpl"/"SMPL_male.npz";config.DIP_DIRECTORY=data/"dip_imu"
        config.TOTALCAPTURE_DIRECTORY=data/"totalcapture";config.TOTALCAPTURE_SMPL_DIRECTORY=data/"totalcapture_smpl"
        config.GREEDY_TOLERANCES_DEG=.1;config.MAIN_TOLERANCE_DEG=None
        config.SETTINGS=replace(config.PILOT,device="cpu",amp=False,model=config.ModelSpec(rnn_hidden=4),
            max_frames_per_sequence=16,window_length=8,stride=8,mic_samples=16,totalcapture_enabled=True,
            tc_subjects=("s1",),swap_max_accepted=1,
            pose_reference_label="sip_dip",random_pose_examples=1,dpi=72)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
        return 0 if result.wasSuccessful() else 1


if __name__=="__main__":raise SystemExit(main())
