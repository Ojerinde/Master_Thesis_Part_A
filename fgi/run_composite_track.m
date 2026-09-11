function out = run_composite_track(name, spoof, varargin)
% RUN_COMPOSITE_TRACK  One end-to-end evaluation of a spoofer parameter set.
%
% Synthesizes the counterfeit waveform, sums it into the authentic TEXBAT
% window, tracks the composite with FGI-GSRx, and exports the resulting
% observables. This is the single unit of work for the waveform-level attack:
% the observables it returns were produced by a receiver tracking a signal a
% transmitter could emit, so no realizability constraint has to be asserted
% afterwards.
%
% Usage
%   out = run_composite_track('probe', struct('alphaDb',3,'dtauChipS',0.2))
%   out = run_composite_track('a06_d10', sp, 'tStart',60, 'tDur',20)
%
% The composite sample file is deleted after tracking unless 'keepBin' is set,
% since each window costs about 100 MB per second of data.

p = inputParser;
addParameter(p, 'authFile', ['D:\BEIHANG UNIVERSITY\Research\code\' ...
    'gnss_adversarial_research\data\raw\texbat\cleanStatic.bin']);
addParameter(p, 'trackMat', ['D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\' ...
    'out\trackData_cleanStatic_full.mat']);
addParameter(p, 'workDir',  'D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\composite');
addParameter(p, 'tStart',   60);
addParameter(p, 'tDur',     20);
% GSRx reads a correlation block past the last processed epoch, and channels sit
% at different code phases, so the sample file must run past msToProcess.
addParameter(p, 'tailSec',  2.0);
% Each window is acquired fresh, so the loops carry an acquisition transient the
% corpus rows never had. Measured against authentic cleanStatic, the transient is
% confined to the first second: from t = 1 s the lock indicators sit within 0.002
% of the corpus. Dropping a warmup window removes a systematic offset that would
% otherwise separate every composite from the training data for reasons that have
% nothing to do with spoofing.
addParameter(p, 'warmupSec', 1.5);
addParameter(p, 'prns',     []);
addParameter(p, 'agc',      true);
% Nav-stage solution rate (ms between fixes). 1000 = 1 Hz keeps the S4 nav run
% tractable; the default track-only sweep never reaches the nav stage.
addParameter(p, 'navSolPeriod', 1000);
% Satellites acquisition may use. A spoofer covers the visible constellation, so
% for a navigation-domain run this is pinned to the PRNs the spoofer actually
% generates. Leaving it open let the receiver acquire an unspoofed satellite
% (PRN 8), which is inconsistent with the commonly-pulled spoofed set and made
% the position solution diverge (residuals of 20-166 km, no valid fix).
addParameter(p, 'acqList', []);
% Carrier phase is reconstructed by integrating the tracked carrier frequency, so
% it carries an arbitrary constant and each channel's replica would otherwise sit
% at a random phase against the authentic signal. Probing the correlation over a
% short prefix recovers that constant per channel, which turns phi0 into a
% controlled attack parameter: 0 aligns the counterfeit with the authentic
% carrier, pi opposes it.
addParameter(p, 'alignPhase',  true);
addParameter(p, 'probeSec',    0.25);
addParameter(p, 'keepBin',  false);
addParameter(p, 'keepMat',  false);
parse(p, varargin{:});
a = p.Results;

fgiRoot = 'D:\BEIHANG UNIVERSITY\Research\code\FGI-GSRx';
addpath(genpath(fgiRoot));
if ~isfolder(a.workDir), mkdir(a.workDir); end

binFile = fullfile(a.workDir, [name '.bin']);
matFile = fullfile(a.workDir, ['trackData_' name '.mat']);
cfgFile = fullfile(a.workDir, ['user_' name '.txt']);
csvFile = fullfile(a.workDir, [name '.csv']);

% -------------------------------------------------------- carrier phase probe
if a.alignPhase
    pr = make_spoof_composite(struct( ...
        'authFile', a.authFile, 'trackMat', a.trackMat, 'selfTest', true, ...
        'tStart', a.tStart, 'tDur', a.probeSec, 'prns', a.prns, ...
        'spoof', struct('alphaDb', 0)));
    % phi0 requested by the caller is measured from the authentic carrier, so
    % the recovered offset is added to it rather than replacing it.
    phi0 = 0;
    if isfield(spoof,'phi0Rad'), phi0 = spoof.phi0Rad; end
    spoof.phi0Rad = phi0 + pr.phaseErr;
    % Pin the channel set to the probe's, so the per-channel phase vector cannot
    % be misaligned by a channel appearing in one window and not the other.
    a.prns = pr.prns;
    fprintf('[%s] carrier phase probe over %.2f s: code alignment %.2f-%.2f\n', ...
            name, a.probeSec, min(pr.corr), max(pr.corr));
end

% ------------------------------------------------------------------ synthesis
tSyn = tic;
syn = make_spoof_composite(struct( ...
    'authFile', a.authFile, 'trackMat', a.trackMat, 'outFile', binFile, ...
    'tStart', a.tStart, 'tDur', a.tDur + a.tailSec, 'prns', a.prns, ...
    'agc', a.agc, 'spoof', spoof));
tSyn = toc(tSyn);
fprintf('[%s] synthesized %.1f s over %d PRNs in %.0f s (agc %+.2f dB, %d saturated)\n', ...
        name, a.tDur, numel(syn.prns), tSyn, syn.agcDb, syn.saturated);

% ------------------------------------------------------------------- tracking
acqList = a.acqList; if isempty(acqList), acqList = a.prns; end
writeConfig(cfgFile, binFile, matFile, a.tDur, fgiRoot, a.workDir, a.navSolPeriod, acqList);
tTrk = tic;
% gsrx saves trackData before the navigation stage, so a failure past that point
% costs nothing here. Frame decoding needs 30 s of data to reach a subframe and
% throws on shorter windows; the library is track-only, exactly as the Paper-1
% corpus is, and the navigation stage is run separately on long windows in S4.
try
    gsrx(cfgFile);
catch ME
    if ~isfile(matFile)
        rethrow(ME);
    end
    fprintf('[%s] post-tracking stage stopped (%s); trackData is complete\n', ...
            name, ME.identifier);
end
tTrk = toc(tTrk);
fprintf('[%s] tracked in %.0f s\n', name, tTrk);

% --------------------------------------------------------------------- export
% The window is entirely post-takeover, so every retained epoch carries the spoof
% label. Setting onsetHi to the warmup drops the acquisition transient through
% the exporter's existing transition band.
job = {struct('trackMat', matFile, 'scenario', name, ...
              'sourceFile', [name '.bin'], 'spoofType', 'synth_waveform', ...
              'onsetLo', -inf, 'onsetHi', a.warmupSec, 'decim', 50)};
export_texbat_track(job, csvFile);

out = struct('name', name, 'spoof', spoof, 'syn', syn, 'csv', csvFile, ...
             'tSynth', tSyn, 'tTrack', tTrk, 'tStart', a.tStart, 'tDur', a.tDur);

% Release any RF file handle GSRx left open (its caught frame-decode error skips
% the fclose), otherwise delete() silently fails and the 1.2 GB composite leaks.
% This is why an earlier full sweep filled the disk. fclose('all') is safe in a
% batch worker; it also closes the config handle already closed above.
fclose('all');
if ~a.keepBin && isfile(binFile)
    delete(binFile);
    if isfile(binFile)
        warning('run_composite_track:leak', 'could not delete %s', binFile);
    end
end
if ~a.keepMat && isfile(matFile), delete(matFile); end
end

% =============================================================================
function writeConfig(cfgFile, binFile, matFile, tDur, fgiRoot, workDir, navSolPeriod, acqList)
% FGI-GSRx user parameter file for one composite window. Front end and loop
% tuning are copied from the TEXBAT configuration so the composite is tracked
% under exactly the settings that produced the corpus.
L = {
 'sys,enabledSignals,[{[''gpsl1'']}],'
 sprintf('sys,msToProcess,%d,', round(tDur*1000))
 'sys,msToSkip,0,'
 'sys,loadDataFile,false,'
 'sys,dataFileIn,'''','
 'sys,saveDataFile,true,'
 sprintf('sys,dataFileOut,''%s'',', matFile)
 'sys,loadIONMetaDataReading,false,'
 'sys,metaDataFileIn,'''''
 'sys,plotSpectra,false,'
 'sys,plotAcquisition,false,'
 'sys,plotTracking,false,'
 'sys,showTrackingOutput,false,'
 'sys,parallelChannelTracking,false,'
 'sys,PCTenabled,false,'
 sprintf('sys,currentWorkingDirectoryForFGIGSRx,''%s\'',', fgiRoot)
 sprintf('sys,trackDataFilePath,''%s\'',', workDir)
 'sys,enableMultiCorrelatorTracking,false,'
 sprintf('nav,navSolPeriod,%d,', navSolPeriod)
 'nav,elevationMask,5,'
 'nav,snrMask,30,'
 'nav,gpsLeapSecond,16,'
 'nav,trueLat,30.2894,'
 'nav,trueLong,-97.7361,'
 'nav,trueHeight,165.0,'
 'gpsl1,frontEnd,''TEXBAT'','
 sprintf('gpsl1,rfFileName,''%s'',', binFile)
 'gpsl1,centerFrequency,1575420000,'
 'gpsl1,samplingFreq,25000000,'
 'gpsl1,bandWidth,2000000,'
 'gpsl1,sampleSize,32,'
 'gpsl1,complexData,true,'
 'gpsl1,iqSwap,false,'
 'gpsl1,clockOffset,0,'
 acqLine(acqList)
 'gpsl1,nonCohIntNumber,2,'
 'gpsl1,cohIntNumber,2,'
 'gpsl1,acqThreshold,10,'
 'gpsl1,maxSearchFreq,7000,'
 'gpsl1,fllNoiseBandwidthWide,200,'
 'gpsl1,fllNoiseBandwidthNarrow,100,'
 'gpsl1,fllNoiseBandwidthVeryNarrow,5,'
 'gpsl1,fllDampingRatio,0.7,'
 'gpsl1,fllLoopGain,1.5,'
 'gpsl1,pllNoiseBandwidthWide,15,'
 'gpsl1,pllNoiseBandwidthNarrow,15,'
 'gpsl1,pllNoiseBandwidthVeryNarrow,10,'
 'gpsl1,pllDampingRatio,0.7,'
 'gpsl1,pllLoopGain,0.1,'
 'gpsl1,dllDampingRatio,0.7,'
 'gpsl1,dllNoiseBandwidth,1,'
 'gpsl1,Nc,0.001,'
 'gpsl1,corrFingers,[-2 -0.25 0 0.25],'
 'gpsl1,earlyFingerIndex,2,'
 'gpsl1,promptFingerIndex,3,'
 'gpsl1,lateFingerIndex,4,'
 'gpsl1,noiseFingerIndex,1,'
 'gpsl1,pllWideBandLockIndicatorThreshold,0.5,'
 'gpsl1,pllNarrowBandLockIndicatorThreshold,0.8,'
 'gpsl1,runningAvgWindowForLockDetectorInMs,20,'
 'gpsl1,fllWideBandLockIndicatorThreshold,0.5,'
 'gpsl1,fllNarrowBandLockIndicatorThreshold,0.7,'
 'gpsl1,enableIonoCorrections,true,'
 'gpsl1,enableTropoCorrections,true,'
 'gpsl1,ionomodel,''default'','
 'gpsl1,ionexFile,'''''
};
fid = fopen(cfgFile, 'w');
if fid < 0, error('run_composite_track:cfg','cannot write %s', cfgFile); end
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); end
fclose(fid);
end

% =============================================================================
function s = acqLine(acqList)
% Acquisition list: all PRNs when unconstrained, otherwise only those the
% spoofer generates, so the navigation solution is not corrupted by an
% authentic satellite the counterfeit constellation does not cover.
if isempty(acqList)
    s = 'gpsl1,acqSatelliteList,[1:32],';
else
    s = sprintf('gpsl1,acqSatelliteList,[%s],', strjoin(string(acqList(:)'), ' '));
end
end
