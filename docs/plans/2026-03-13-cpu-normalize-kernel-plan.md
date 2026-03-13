#CPUNormalizeKernelImplementationPlan

>**ForClaude:**REQUIREDSUB-SKILL:Use`superpowers:executing-plans`toimplementthisplantask-by-task.

**Goal:**Route`torch.nn.functional.normalize`throughtheregisteredbackend`normalize`opandaddfocusedCPU/contractteststhatlockdownforwardparityandkeyargumentbehavioronCPU.

**Architecture:**KeepthisbatchtightlyscopedtothepublicfunctionalAPIpathplusfocusedtests.Thebackendkernel,schema,andCPUregistrationalreadyexist;thisbatchshouldprovethepublicAPIreachesthatkernelpathandpreservesPyTorch-alignedbehaviorforcommonforwardcasesandbasicargumenthandling.

**TechStack:**Python,pytest,Candledispatchstack,CPUbackendkernel,PyTorchforwardparityreference.

---

###Task1:AddfocusedCPUregressiontestsfor`F.normalize`

**Files:**
-Modify:`tests/cpu/test_nn_functional.py`

**Step1:Writethefailingtests**

Addfocusedtestsfor:
-`torch.nn.functional.normalize(x,p=2.0,dim=1)`matchesPyTorchonasimple2Dinput
-`torch.nn.functional.normalize(x,p=1.0,dim=0)`matchesPyTorchonanon-defaultnorm/dim
-zero-normrowsrespect`eps`behaviorandmatchPyTorchnumerically
-aroutingtestthatproves`F.normalize(...)`goesthrough`dispatch("normalize",...)`

**Step2:RunthetargetedtestsliceandverifyRED**

Run:

```bash
PYTHONPATH=srcpytesttests/cpu/test_nn_functional.py-k"normalize_matches_torchornormalize_zero_normornormalize_routes_through_dispatch"-v--tb=short
```

Expected:thedispatch-routingtestfailsuntil`F.normalize`stopsusingthecompositepath.

---

###Task2:Addonefocusedcontractparitytestfornormalize

**Files:**
-Create:`tests/contract/test_training_core_normalize_parity.py`

**Step1:Writeafocusedcontracttest**

Usetheexistingtraining-coreparityharnessstyletocoveronestableforwardparitycaseagainstrealPyTorch.

Suggestedcoverage:
-forwardvalue/dtype/shapeparityfor`F.normalize(input,p=2.0,dim=1,eps=1e-12)`

**Step2:Runthetargetedcontracttest**

Run:

```bash
PYTHONPATH=srcpytesttests/contract/test_training_core_normalize_parity.py-v--tb=short
```

Expected:passoncethepublicpathandparitytestarebothcorrect.

---

###Task3:Route`F.normalize`throughdispatch

**Files:**
-Modify:`src/candle/nn/functional.py`

**Step1:Replacethecompositeimplementationwithdispatch**

Update`normalize()`tocalltheregisteredopdirectly:

```python
from.._dispatchimportdispatch
returndispatch("normalize",input.device.type,input,p,dim,eps)
```

KeepthepublicAPIsurfaceunchanged.

**Step2:Re-runthetargetedCPUandcontracttests**

Run:

```bash
PYTHONPATH=srcpytesttests/cpu/test_nn_functional.py-k"normalize_matches_torchornormalize_zero_normornormalize_routes_through_dispatch"-v--tb=short
PYTHONPATH=srcpytesttests/contract/test_training_core_normalize_parity.py-v--tb=short
```

Expected:bothtargetedslicespass.

---

###Task4:Runadjacentregressioncoverage

**Step1:RunthefullfunctionalCPUfile**

Run:

```bash
PYTHONPATH=srcpytesttests/cpu/test_nn_functional.py-v--tb=short
```

**Step2:Runtherequiredcontractgate**

Run:

```bash
PYTHONPATH=srcpytesttests/contract/-v--tb=short
```

**Step3:RunthebroaderCPUgate**

Run:

```bash
PYTHONPATH=srcpytesttests/cpu/-v--tb=short
```

---

###Task5:Committhebatchcleanly

**Files:**
-Modify:`src/candle/nn/functional.py`
-Modify:`tests/cpu/test_nn_functional.py`
-Create:`tests/contract/test_training_core_normalize_parity.py`
-Add:`docs/plans/2026-03-13-cpu-normalize-kernel-plan.md`

**Step1:Inspectdiff**

Run:

```bash
gitstatus--short
gitdiff--src/candle/nn/functional.pytests/cpu/test_nn_functional.pytests/contract/test_training_core_normalize_parity.pydocs/plans/2026-03-13-cpu-normalize-kernel-plan.md
```

**Step2:Commit**

```bash
gitaddsrc/candle/nn/functional.pytests/cpu/test_nn_functional.pytests/contract/test_training_core_normalize_parity.pydocs/plans/2026-03-13-cpu-normalize-kernel-plan.md
gitcommit-m"fix:routenormalizethroughbackendkernel"
```
