#MoveaxisMovedimParityImplementationPlan

>**ForClaude:**REQUIREDSUB-SKILL:Usesuperpowers:executing-planstoimplementthisplantask-by-task.

**Goal:**ClosetheremainingCPU-side`moveaxis`/`movedim`paritygapsbyaddingfocusedcoveragefornormalbehavior,aligningschema-levelinvalid-dimensionerrorswithPyTorch,andfixingincorrectpendingshapesinpipeline/metamode.

**Architecture:**Keepthisbatchmechanism-focused.ReusetheexistingpublicAPI,schemas,CPUkernels,andautogradregistrations.AddredtestsfirstinCPU,contract,andpipelinesuites;thenmakethesmallestpossiblechangesinschemavalidationandmetainferencesoruntimebehaviorandpending-shapemetadatamatchPyTorchsemantics.

**TechStack:**Python,pytest,Candledispatch/schema/metasystem,NumPy-backedCPUkernels.

---

###Task1:AddfocusedCPUtop-levelparitytests

**Files:**
-Modify:`tests/cpu/test_top_level_ops.py`
-Test:`tests/cpu/test_top_level_ops.py`

**Step1:Writethefailingtest**

Addteststhatcover:
-`torch.movedim(x,0,2)`returnsshape`(3,4,2)`withexpectedvalues.
-`torch.moveaxis(x,0,2)`matches`torch.movedim(x,0,2)`exactly.
-`x.movedim((0,2),(2,0))`returnsshape`(4,3,2)`withexpectedvalues.
-`x.moveaxis((0,2),(2,0))`matches`x.movedim((0,2),(2,0))`exactly.

**Step2:Runtesttoverifycurrentstatus**

Run:`PYTHONPATH=srcpytesttests/cpu/test_top_level_ops.py-k"moveaxisormovedim"-v--tb=short`
Expected:passorexposeanyunexpectedfunctionalgap.

**Step3:Keeporrefineonlyifneeded**

Iffunctionalcoverageisgreenimmediately,keepthetestsasregressioncoverageandmoveon.

**Step4:Re-runtest**

Runthesamecommandandconfirmcleanoutput.

**Step5:Commit**

Donotcommityet;batchcommitafterschema/metafixesaregreen.

###Task2:Addfailingpipeline/metaregressiontest

**Files:**
-Modify:`tests/cpu/test_pipeline.py`
-Test:`tests/cpu/test_pipeline.py`

**Step1:Writethefailingtest**

Addafocusedtestthatexecutesunder`withtorch.pipeline():`andassertspendingtensorshapesarealreadypermutedcorrectlyfor:
-`torch.movedim(x,0,2)`->pendingshape`(3,4,2)`
-`torch.moveaxis(x,(0,2),(2,0))`->pendingshape`(4,3,2)`

Alsoasserttheshapesremainthesameafterflush.

**Step2:Runtesttoverifyitfails**

Run:`PYTHONPATH=srcpytesttests/cpu/test_pipeline.py-k"movedimormoveaxis"-v--tb=short`
Expected:FAILbecause`infer_movedim`currentlyreturnstheinputshapeunchanged.

**Step3:Writeminimalimplementation**

Update`src/candle/_backends/meta/infer.py`so`infer_movedim`computesthepermutedoutputshapeandcontiguousstride,withsource/destinationnormalizationmatchingruntimesemantics.

**Step4:Runtesttoverifyitpasses**

RunthesamecommandandconfirmPASS.

**Step5:Commit**

Donotcommityet;batchcommitaftercontractparityisgreen.

###Task3:Addfailingschema-levelerror-contracttests

**Files:**
-Modify:`tests/contract/test_schema_dim_validation.py`
-Modify:`src/candle/_dispatch/schema.py`
-Test:`tests/contract/test_schema_dim_validation.py`

**Step1:Writethefailingtest**

AddexactTorch-alignmenttestsusing`assert_torch_error(...)`forboth`dispatch("movedim",...)`and`dispatch("moveaxis",...)`covering:
-duplicatesourcedims:`([0,0],[1,2])`
-source/destinationlengthmismatch:`([0,1],[2])`
-out-of-rangedim:`(3,0)`onarank-3tensor

**Step2:Runtesttoverifyitfails**

Run:`PYTHONPATH=srcpytesttests/contract/test_schema_dim_validation.py-k"movedimormoveaxis"-v--tb=short`
Expected:FAILbecausecurrentbehaviorleaksNumPy/backendexceptionsandtexts.

**Step3:Writeminimalimplementation**

In`src/candle/_dispatch/schema.py`:
-addavalidatorfor`movedim`/`moveaxis`sourceanddestinationarguments
-normalizeint/list/tupleformsintocomparabledimlists
-rejectboolsandnon-intentrieswithTorch-styletuple-of-intserrorswhereapplicable
-rejectduplicatenormalizeddimswithTorch-style`movedim:repeateddimin\`source\``/`\`destination\``messages
-rejectsource/destinationlengthmismatchwithTorch-style`Invalidsourceordestinationdims`message
-rejectout-of-rangedimswiththeexistingTorch-style`Dimensionoutofrange...`text

Keepthevalidatorscopedonlytotheseops.

**Step4:Runtesttoverifyitpasses**

RunthesamecommandandconfirmPASS.

**Step5:Commit**

Donotcommityet;batchcommitafterbroaderverification.

###Task4:Runfocusedandbroaderverification

**Files:**
-Verify:`tests/cpu/test_top_level_ops.py`
-Verify:`tests/cpu/test_pipeline.py`
-Verify:`tests/contract/test_schema_dim_validation.py`
-Verify:`tests/contract/`
-Verify:`tests/cpu/`

**Step1:Runfocusedregressioncommands**

Run:
-`PYTHONPATH=srcpytesttests/cpu/test_top_level_ops.py-k"moveaxisormovedim"-v--tb=short`
-`PYTHONPATH=srcpytesttests/cpu/test_pipeline.py-k"movedimormoveaxis"-v--tb=short`
-`PYTHONPATH=srcpytesttests/contract/test_schema_dim_validation.py-k"movedimormoveaxis"-v--tb=short`

**Step2:Runrequiredcontractgate**

Run:`PYTHONPATH=srcpytesttests/contract/-v--tb=short`
Expected:PASS

**Step3:RunbroaderCPUgate**

Run:`PYTHONPATH=srcpytesttests/cpu/tests/contract/-v--tb=short`
Expected:PASS

**Step4:Commit**

```bash
gitadddocs/plans/2026-03-14-cpu-moveaxis-movedim-parity-plan.md\
tests/cpu/test_top_level_ops.py\
tests/cpu/test_pipeline.py\
tests/contract/test_schema_dim_validation.py\
src/candle/_dispatch/schema.py\
src/candle/_backends/meta/infer.py
gitcommit-m"fix:alignmoveaxismovedimparity"
```
