#!/usr/bin/env python3
"""Repair existing FP4-G16 base slots with TRAIN CE/KL; no Resurface."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import (runtime, calibration, evaluation, fp4,
                          fp4_base_repair as repair, fp4_repaired_checkpoint as package)
from mamba2_recall.fp4_checkpoint import load_packed_model, verify_checkpoint
from mamba2_recall.resurface_loss import staged_loss_backward
from evaluate_resurface_more import pin_replay_backend, check_replay_backend

SCHEDULE_SHA = '6c0e280e623dc5e78ce478bac006b9844d73a730b18f03a34b70649f7a2eae07'
PROSE_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
TOKENS_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
MANIFEST_SHA = '4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07'
LEDGER_SHA = '55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b'
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
STEPS = (224, 448, 672, 896)
HP = dict(successful_updates=896, epochs=2, windows_per_epoch=448,
          window_tokens=2048, targets_per_update=2047, seed=2026093001,
          optimizer='AdamW', learning_rate=1e-4, betas=[.9,.999], epsilon=1e-8,
          weight_decay=0., gradient_clip_norm=1., ce_coefficient=1.,
          teacher_to_student_kl_coefficient=.1, temperature=1., logits_chunk_tokens=64,
          cosine_multiplier_start=1., cosine_multiplier_end=.1,
          initial_loss_scale=1024., maximum_overflow_retries=8,
          candidate_successful_updates=list(STEPS), selected_tensors=393,
          selected_parameters=3580928, selected_fp16_bytes=7161856)

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def put(path, value, fresh=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fresh: need(not path.exists(), 'Existing evidence: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu_tree(v) for v in value)
    return value

def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def codes():
    result={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.is_file() and path.suffix=='.py' and path.is_relative_to(ROOT):
            result[str(path.relative_to(ROOT))]=sha(path)
    result[str(Path(__file__).resolve().relative_to(ROOT))]=sha(__file__)
    return dict(sorted(result.items()))

def snapshot(model):
    return {n:(id(m),getattr(m.forward,'__func__',m.forward),dict(m._forward_pre_hooks),dict(m._forward_hooks))
            for n,m in model.named_modules()}

def restored(model, old):
    need(snapshot(model)==old,'Native methods/hooks changed')
    return dict(complete=True,forward_methods_and_hooks_exact=True,modules=len(old))

def hashes(model):return {n:repair.tensor_hash(p) for n,p in model.named_parameters()}

def grads(bank):
    need(all(p.grad is not None for p in bank.parameters()),'Missing selected gradient')
    return dict(finite=all(bool(torch.isfinite(p.grad).all()) for p in bank.parameters()),
                nonzero=sum(int(torch.count_nonzero(p.grad)) for p in bank.parameters()),
                tensors=len(bank.masters),parameters=sum(p.numel() for p in bank.parameters()))

def parity(bank, tokens):
    return [bank.export_parity(tokens[0,None,:length].cuda()) for length in (128,512)]

def step(bank, teacher, window, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    ids=window[None,:-1].cuda();targets=window[None,1:].cuda()
    with torch.no_grad():teacher_hidden=teacher.backbone(ids)
    hidden=bank.forward(ids,use_checkpoint=True)
    scale=scaler.get_scale()
    row=staged_loss_backward(hidden,targets,bank.model.lm_head,scaler=scaler,
          teacher_hidden=teacher_hidden,teacher_head=teacher.lm_head,
          ce_weight=1.,kl_weight=.1,chunk_tokens=64)
    del hidden,teacher_hidden,ids,targets
    scaler.unscale_(optimizer);gradient=grads(bank)
    overflow=not gradient['finite'] or not row['scaled_hidden_gradient_finite']
    if overflow:
        before=(digest(bank.state_dict()),digest(optimizer.state_dict()))
        scaler.update(new_scale=scale*.5)
        need(before==(digest(bank.state_dict()),digest(optimizer.state_dict())),'Overflow mutated parameters/optimizer')
        norm=None
    else:
        need(gradient['nonzero']>0,'All selected gradients are zero')
        norm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update();bank.stability_receipt()
    bank.assert_frozen();optimizer.zero_grad(set_to_none=True)
    return dict(**row,overflow=overflow,gradients=gradient,loss_scale_before=scale,
                loss_scale_after=scaler.get_scale(),gradient_norm_before_clip=norm,
                overflow_optimizer_and_masters_unchanged=True if overflow else None)

def smoke_gradient(bank, tokens):
    """Verify altered-master checkpoint replay against ordinary native autograd."""
    initial=bank.state_dict();ids=tokens[0,None,:128].cuda();results=[];hidden_hashes=[]
    try:
        with torch.no_grad():
            for p in bank.parameters():p.add_(torch.linspace(-.003,.003,p.numel(),device=p.device).reshape(p.shape))
        for enabled in (False,True):
            for p in bank.parameters():p.grad=None
            hidden=bank.forward(ids,use_checkpoint=enabled);hidden_hashes.append(repair.tensor_hash(hidden))
            tangent=torch.linspace(-1,1,hidden.shape[-1],device=hidden.device)
            (1024*(hidden.float()*tangent).mean()).backward()
            need(grads(bank)['finite'] and grads(bank)['nonzero']>0,'Invalid smoke gradients')
            results.append({n:p.grad.detach().cpu().clone() for n,p in bank.masters.items()})
            del hidden
        need(hidden_hashes[0]==hidden_hashes[1],'Checkpoint forward mismatch')
        maximum=0.
        for name in results[0]:
            left,right=results[0][name],results[1][name]
            need(torch.allclose(left,right,rtol=.005,atol=.00003),'Checkpoint gradient mismatch: '+name)
            maximum=max(maximum,float((left-right).abs().max()))
        return dict(complete=True,passed=True,tensors=len(results[0]),rtol=.005,atol=.00003,
                    maximum_absolute_difference=maximum,hidden_bitwise_equal=True,
                    perturbed_masters=True,optimizer_steps_performed=0)
    finally:
        bank.load_state_dict(initial)
        for p in bank.parameters():p.grad=None
        bank.assert_frozen()

def checkpoint_candidate(bank, report, args, tokens, optimizer, scaler):
    successful=report['successful_updates'];directory=args.out_dir/f'step{successful:04d}'
    directory.mkdir();retained=bank.export_fp16();export=directory/'retained_fp16.pt'
    torch.save(retained,export)
    master_path=directory/'optimizer_checkpoint.pt'
    torch.save(dict(masters=bank.state_dict(),optimizer=cpu_tree(optimizer.state_dict()),
                    scaler=scaler.state_dict(),successful_updates=successful,attempts=report['attempts']),master_path)
    saved=torch.load(master_path,map_location='cpu',weights_only=True)
    replay_retained=torch.load(export,map_location='cpu',weights_only=True)
    need(set(saved['masters'])==set(replay_retained)==set(bank.masters),'Checkpoint readback inventory differs')
    need(all(torch.equal(saved['masters'][n].half(),replay_retained[n]) and
             torch.equal(retained[n],replay_retained[n]) for n in retained),
         'Actual FP16 export differs from saved FP32 master casts')
    need(digest(saved['masters'])==digest(bank.state_dict()) and
         digest(saved['optimizer'])==digest(optimizer.state_dict()) and
         digest(saved['scaler'])==digest(scaler.state_dict()) and
         saved['successful_updates']==successful and saved['attempts']==report['attempts'],
         'Master/optimizer/scaler/counter checkpoint readback differs')
    readback=dict(complete=True,passed=True,retained_tensors=393,
                  fp16_export_equals_saved_master_cast=True,masters_exact=True,
                  optimizer_exact=True,scaler_exact=True,counters_exact=True)
    native_parity=parity(bank,tokens)
    core=dict(format='FP4_BASE_REPAIR_CHECKPOINT_CORE_V1',complete=True,mode=report['mode'],
              discarded=report['mode']=='smoke',successful_updates=successful,attempts=report['attempts'],
              binding=report['binding'],history=report['history'],no_resurface_installed=True,
              retained_export_file=export.name,retained_export_sha256=sha(export),
              retained_tensor_sha256={n:repair.tensor_hash(v) for n,v in retained.items()},
              optimizer_checkpoint=dict(file=master_path.name,sha256=sha(master_path),bytes=master_path.stat().st_size),
              checkpoint_readback=readback,
              parent_base_frozen=bank.assert_frozen(),native_export_parity=native_parity,
              stability=bank.stability_receipt())
    core_path=directory/'report.json';put(core_path,core,fresh=True)
    bound=dict(protocol_sha256=report['binding']['protocol_sha256'],training_report_path=str(core_path),
               training_report_sha256=sha(core_path),successful_updates=successful,
               discarded=report['mode']=='smoke')
    manifest=package.export_repaired_checkpoint(args.packed_dir,retained,directory/'model',bound)
    return directory,retained,manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packed-dir','source-dir','prose-tokens','schedule','out-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/FP4_BASE_REPAIR_V1_PROTOCOL.md')
    parser.add_argument('--expected-protocol-sha256',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-report',type=Path)
    args=parser.parse_args()
    need(not args.out_dir.exists(),'Fresh output directory required')
    need(sha(args.protocol)==args.expected_protocol_sha256,'Frozen protocol differs')
    need(sha(args.schedule)==SCHEDULE_SHA and sha(args.prose_tokens)==TOKENS_SHA,'Pinned TRAIN files differ')
    prose_path=ROOT/'docs/prose_train_manifest.json'
    need(sha(prose_path)==PROSE_SHA,'Pinned TRAIN manifest differs')
    prose=read(prose_path);schedule=read(args.schedule)
    need(prose['split']=='train' and prose['evaluation_data_used'] is False and
         prose['heldout_used_for_fitting'] is False,'Non-TRAIN fitting data')
    generator=torch.Generator(device='cpu').manual_seed(2026093001)
    order=[i for _ in range(2) for i in torch.randperm(448,generator=generator).tolist()]
    need(schedule['order']==order and schedule['seed']==2026093001,'Fixed TRAIN schedule differs')
    tokens=torch.load(args.prose_tokens,map_location='cpu',weights_only=True)
    need(tokens.dtype==torch.int64 and list(tokens.shape)==[448,2048] and
         runtime.token_digest(tokens.flatten().numpy())==prose['training_tokens_sha256_int64le'],
         'Pinned full TRAIN token geometry/content differs')
    need(sha(args.packed_dir/'weight_manifest.json')==MANIFEST_SHA and
         sha(args.packed_dir/'conversion_receipt.json')==LEDGER_SHA,'Parent packed identity differs')
    parent,ledger=verify_checkpoint(args.packed_dir)
    need(torch.cuda.is_available(),'RTX GPU required')
    torch.set_num_threads(8);torch.manual_seed(2026093001)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    backend=pin_replay_backend();source_hashes=codes()
    binding=dict(protocol_sha256=sha(args.protocol),parent_manifest_sha256=MANIFEST_SHA,
      parent_conversion_receipt_sha256=LEDGER_SHA,source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
      tokenizer_sha256=runtime.TOKENIZER_SHA256,schedule_sha256=SCHEDULE_SHA,
      prose_manifest_sha256=PROSE_SHA,prose_tokens_sha256=TOKENS_SHA,
      code_sha256=source_hashes,hyperparameters=HP,backend_policy=backend,
      training_split='train',state_mode='fp16_native_ssd',no_resurface_installed=True)
    if not args.smoke:
        need(args.smoke_report is not None,'Completed discarded smoke required')
        smoke=read(args.smoke_report)
        need(smoke['complete'] is True and smoke['mode']=='smoke' and smoke['successful_updates']==1
             and smoke['binding']==binding,'Smoke is incomplete or bound to different code/recipe')
    args.out_dir.mkdir(parents=True);report_path=args.out_dir/'report.json'
    report=dict(format='FP4_BASE_REPAIR_TRAIN_V1',complete=False,mode='smoke' if args.smoke else 'formal',
                started_unix=time.time(),binding=binding,successful_updates=0,attempts=0,overflows=0,
                history=[],candidates=[],smoke_report_sha256=sha(args.smoke_report) if args.smoke_report else None,
                environment=runtime.environment_receipt(),no_resurface_installed=True,
                no_statequant_controller_installed=True,test_used_for_training_or_selection=False,
                backend_policy=backend)
    put(report_path,report)
    try:
        print('[load] strict packed FP4 base and separate original FP16 teacher',flush=True)
        student=load_packed_model(args.packed_dir,device='cuda');teacher=runtime.load_source_model(args.source_dir)
        student_snapshot=snapshot(student);teacher_snapshot=snapshot(teacher)
        teacher_hashes=hashes(teacher);bank=repair.SmallMasters(student,ledger)
        need(teacher_hashes=={n:v['source_fp16_sha256'] for n,v in ledger['tensors'].items()},
             'Original FP16 teacher content differs from pinned original-source ledger')
        teacher_pointers={p.data_ptr() for p in teacher.parameters()}
        need(all(p.data_ptr() not in teacher_pointers for p in student.parameters()),'Teacher/student storage aliases')
        report.update(initial_base_content=bank.assert_frozen(check_values=True),
                      teacher_initial_tensor_sha256=teacher_hashes,teacher_student_independent_storage=True,
                      initial_native_parity=parity(bank,tokens),selected_inventory=bank.inventory,
                      selected_parameters=3580928,retained_fp16_payload_bytes=7161856)
        tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        validation_ids,validation_data=calibration.load_wikitext_tokens(tokenizer,'validation',REVISION)
        validation_windows=evaluation.ppl_windows(validation_ids,2048)
        need(len(validation_windows)==130 and sum(len(w)-1 for _,w in validation_windows)==264764,
             'Incomplete full validation population')
        raw=validation_ids.numpy().astype('<i8',copy=False).tobytes()
        (args.out_dir/'validation_tokens.int64le').write_bytes(raw)
        report['validation_dataset']=validation_data
        if args.smoke:report['checkpoint_gradient_smoke']=smoke_gradient(bank,tokens)
        else:
            report['baseline_validation']=evaluation.evaluate_ppl(student,validation_windows,
                                          execution='prefill',logits_chunk=64)
        optimizer=torch.optim.AdamW(bank.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.,growth_interval=2000)
        limit=1 if args.smoke else 896
        put(report_path,report)
        while report['successful_updates']<limit:
            successful=report['successful_updates'];window_index=order[successful]
            learning_rate=1e-4*(.1+.9*.5*(1+math.cos(math.pi*successful/895)))
            for group in optimizer.param_groups:group['lr']=learning_rate
            started=time.time();values=step(bank,teacher,tokens[window_index],optimizer,scaler)
            report['attempts']+=1
            if values['overflow']:report['overflows']+=1
            else:report['successful_updates']+=1
            row=dict(attempt=report['attempts'],successful_updates_before=successful,
                     successful_updates_after=report['successful_updates'],window_index=window_index,
                     epoch=successful//448,position_in_epoch=successful%448,
                     learning_rate=learning_rate,seconds=time.time()-started,**values)
            report['history'].append(row)
            need(report['overflows']<=8,'Exceeded frozen eight-overflow budget')
            if report['attempts']%16==0 or values['overflow']:
                put(report_path,report)
                print(f"[train] {report['successful_updates']}/{limit}, attempts={report['attempts']}, overflows={report['overflows']}",flush=True)
            if not values['overflow'] and (args.smoke or report['successful_updates'] in STEPS):
                print('[export] step '+str(report['successful_updates']),flush=True)
                directory,retained,manifest=checkpoint_candidate(bank,report,args,tokens,optimizer,scaler)
                candidate_dir=directory/'model';candidate_sha=sha(candidate_dir/'weight_manifest.json')
                if args.smoke:
                    # Exercise the standalone file decoder, rather than only in-memory folding.
                    replay=package.load_repaired_model(candidate_dir,device='cuda')
                    with torch.no_grad():
                        ids=tokens[0,None,:128].cuda();expected=bank.forward(ids,use_checkpoint=False)
                        actual=replay.backbone(ids)
                        need(torch.equal(expected,actual),'Actual repaired-checkpoint native reload differs')
                        need(torch.equal(student.lm_head(expected[:,-8:]),replay.lm_head(actual[:,-8:])),
                             'Actual repaired-checkpoint output reload differs')
                    report['actual_packed_reload_parity']=dict(complete=True,passed=True,input_tokens=128,
                                                               hidden_and_last_eight_logits_bitwise_equal=True)
                    report['smoke_candidate_manifest_sha256']=candidate_sha
                    del replay,expected,actual,ids
                else:
                    print('[validation] step '+str(report['successful_updates']),flush=True)
                    with repair.install_fp16(student,retained,bank.inventory):
                        ppl=evaluation.evaluate_ppl(student,validation_windows,execution='prefill',logits_chunk=64)
                    restoration=restored(student,student_snapshot);bank.assert_frozen()
                    val_path=directory/'validation.json'
                    put(val_path,dict(format='FP4_BASE_REPAIR_VALIDATION_V1',complete=True,
                        successful_updates=report['successful_updates'],candidate_manifest_sha256=candidate_sha,
                        dataset=validation_data,ppl=ppl,backend_policy=backend,
                        no_resurface_installed=True,no_statequant_controller_installed=True,
                        native_restoration=restoration),fresh=True)
                    report['candidates'].append(dict(successful_updates=report['successful_updates'],
                        candidate_dir=str(candidate_dir),candidate_manifest_sha256=candidate_sha,
                        validation_report_path=str(val_path),validation_report_sha256=sha(val_path),
                        validation_nll=ppl['nll'],validation_target_tokens=ppl['target_tokens']))
                    print(f"[validation complete] step={report['successful_updates']}, ppl={ppl['ppl']:.8f}",flush=True)
                put(report_path,report)
        report.update(final_base_content=bank.assert_frozen(check_values=True),
                      final_native_export_parity=parity(bank,tokens),
                      final_native_restoration=restored(student,student_snapshot),
                      teacher_final_tensor_sha256=hashes(teacher),teacher_native_restoration=restored(teacher,teacher_snapshot),
                      final_stability=bank.stability_receipt(),backend_policy_final_check=check_replay_backend(backend),
                      gpu_memory=runtime.gpu_memory_receipt())
        need(report['teacher_final_tensor_sha256']==teacher_hashes,'Original FP16 teacher changed')
        need(codes()==source_hashes and sha(args.protocol)==binding['protocol_sha256'],
             'Bound source/protocol changed during execution')
        if not args.smoke:
            need([v['successful_updates'] for v in report['candidates']]==list(STEPS),'Incomplete fixed validation candidate pool')
        report.update(complete=True,elapsed_seconds=time.time()-report['started_unix'])
        put(report_path,report)
        if not args.smoke:
            chosen=min(report['candidates'],key=lambda x:(x['validation_nll'],x['successful_updates']))
            selection=dict(format='FP4_BASE_REPAIR_SELECTION_V1',complete=True,
                protocol_sha256=binding['protocol_sha256'],training_report_path=str(report_path),
                training_report_sha256=sha(report_path),successful_updates=896,
                test_used_for_selection=False,criterion='minimum_full_validation_pooled_nll',
                tie_breaker='earlier_successful_update',candidate_pool=report['candidates'],
                chosen_successful_updates=chosen['successful_updates'],chosen_candidate_dir=chosen['candidate_dir'],
                chosen_candidate_manifest_sha256=chosen['candidate_manifest_sha256'])
            put(args.out_dir/'selection.json',selection,fresh=True)
        print(json.dumps(dict(complete=True,mode=report['mode'],successful_updates=report['successful_updates'],
                              attempts=report['attempts'],overflows=report['overflows']),indent=2),flush=True)
    except Exception as exc:
        report.update(complete=False,error=str(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.time()-report['started_unix'])
        put(report_path,report);raise

if __name__=='__main__':main()
