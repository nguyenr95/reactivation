function ml = bhv2_to_bhv_RE(MLdata, TrialRecord) %data and TrialRecord are variables from mlconcatenate
ml.ConditionNumber = MLdata.Condition; 
ml.TrialError = MLdata.TrialError; 
ml.TimingFileByCond = TrialRecord.TaskInfo.TimingFileByCond; 
end 