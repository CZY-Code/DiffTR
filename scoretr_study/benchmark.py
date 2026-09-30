"""Check grid equivalence and measure faithful ScoreTR update time."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
import torch, yaml
from EBM.energy4Imgd import ScoreMatching
from grid import BoundedScoreMatching
torch.set_num_threads(1)
torch.manual_seed(3407)
cfg = yaml.full_load((HERE/'config.yaml').read_text())['model']
cfg['h_dim'] = list(cfg['h_dim'])
model = BoundedScoreMatching(tensor_shape=[320,307,191], dtype=torch.FloatTensor,
                            device='cuda:0', **cfg).cuda()
idx = torch.randint(0, 100, (16,3), device='cuda')
x = torch.rand(16, device='cuda')
with torch.no_grad():
    a = ScoreMatching.grid_search_posterior4train(model,idx,x,1/255)
    b = model.grid_search_posterior4train(idx,x,1/255)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    torch.testing.assert_close(a[1],b[1],rtol=1e-5,atol=1e-6)
    predicted = model.predict(idx,x_range=[0.,1.],epsilon=1/255)
    expected = ScoreMatching.grid_search_posterior(model,idx,[0.,1.],1/255)
    torch.testing.assert_close(predicted,expected,rtol=0,atol=0)
torch.manual_seed(12)
la = model.anneal_dsm_Sparse(idx,x,0,100)
la.backward()
ga = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
model.zero_grad(set_to_none=True)
original = model.grid_search_posterior4train
model.grid_search_posterior4train = lambda *a, **kw: ScoreMatching.grid_search_posterior4train(model,*a,**kw)
torch.manual_seed(12)
lb = model.anneal_dsm_Sparse(idx,x,0,100)
lb.backward()
torch.testing.assert_close(la,lb,rtol=1e-5,atol=1e-5)
for p, g in zip(model.parameters(),ga):
    if g is not None: torch.testing.assert_close(p.grad,g,rtol=1e-4,atol=1e-5)
model.grid_search_posterior4train = original
opt = torch.optim.Adam(model.parameters(),lr=.001)
idx = torch.randint(0,100,(8192,3),device='cuda'); x=torch.rand(8192,device='cuda')
times=[]
for i in range(5):
    torch.cuda.synchronize(); start=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss=model.anneal_dsm_Sparse(idx,x,i,100);loss.backward();opt.step()
    torch.cuda.synchronize();times.append(time.perf_counter()-start)
result={'grid_and_loss_and_gradient_checks':'passed','prediction_grid_check':'passed','batch_size':8192,'seconds_per_update':times}
(HERE/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
