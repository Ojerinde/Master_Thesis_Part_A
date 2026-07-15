function export_texbat_track(varargin)
% EXPORT_TEXBAT_TRACK  Paper-1 observable-level feature export from FGI-GSRx
% tracking output (trackData) for the TEXBAT corpus.
%
% Reads ONLY tracking-stage observables (no navigation stage), so it cannot
% trigger the run_fgi_nav save runaway and needs no ephemeris decode. Authentic
% vs spoof is labelled by the documented within-recording spoofing ONSET, so the
% discriminant is the spoofing signal itself, not the recording identity.
%
% Feature columns are the per-epoch fields FGI-GSRx retains for every channel
% (allocateTrackChannel.m): the C/N0 family, Doppler, carrier frequency, prompt
% correlator I/Q (hence prompt power and phase), the DLL discriminator, the FLL
% filter output, and the PLL/FLL lock indicators. These are standard receiver
% observables, replacing the earlier hand-rolled-receiver corpus.
%
% Usage
%   export_texbat_track()              % default jobs, default output CSV
%   export_texbat_track(jobs)          % custom Nx1 cell of job structs
%   export_texbat_track(jobs, outCsv)  % custom jobs + output path
%
% Job struct fields (see mkjob):
%   trackMat   - trackData .mat written by gsrx.m for one TEXBAT scenario
%   scenario   - short tag ('cleanstatic','ds7','ds2','ds3')
%   sourceFile - original .bin name (provenance)
%   spoofType  - 'clean' / 'time_push' / 'position_push'
%   onsetLo    - last genuine second   (t_sec <  onsetLo  -> genuine)
%   onsetHi    - first spoof second    (t_sec >= onsetHi  -> spoof)
%                rows in [onsetLo, onsetHi) are the takeover transition, dropped.
%                onsetLo = inf marks the whole recording genuine (cleanStatic).
%   decim      - keep every decim-th 1 ms epoch (50 -> 20 Hz feature cadence)

fgiRoot = 'D:\BEIHANG UNIVERSITY\Research\FGI-GSRx';
if isfolder(fgiRoot), addpath(genpath(fgiRoot)); end

trackRoot  = 'D:\BEIHANG UNIVERSITY\Research\FGI_Data\out';
dstDefault = ['D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\' ...
              'data\processed\texbat_track_combined.csv'];

% Authoritative TEXBAT onsets (Humphreys, TEXBAT ds7/ds8 doc; ION 2012 [27]):
%   cleanStatic : entirely authentic.
%   ds7         : 0-110 s clean (identical to cleanStatic), 110-150 s carrier-
%                 phase-aligned takeover (dropped), 150-468 s code drift.
%   ds2         : Scenario 2 (10 dB overpowered): takeover ~80 s, time pull-off
%                 ~115 s (ION 2012). genuine t<75 s, spoof t>=125 s.
%   ds3         : Scenario 3 (1.3 dB matched, freq-locked): same timing as ds2.
%                 genuine t<75 s, spoof t>=125 s.
%   ds8         : = ds7 + SCER; same timing (110/150). Near-duplicate of ds7 at
%                 the observable level; optional.
%   ds5         : DYNAMIC (Scenario 5) - EXCLUDED from Paper 1 (static corpus
%                 only, to avoid a static-vs-dynamic motion confound).
defaultJobs = {
  mkjob(fullfile(trackRoot,'trackData_cleanStatic_full.mat'), 'cleanstatic', 'cleanStatic.bin', 'clean',              inf, inf, 50)
  mkjob(fullfile(trackRoot,'trackData_ds2_full.mat'),         'ds2',         'ds2.bin',         'time_push_10db',      75, 125, 50)
  mkjob(fullfile(trackRoot,'trackData_ds3_full.mat'),         'ds3',         'ds3.bin',         'time_push_1p3db',     75, 125, 50)
  mkjob(fullfile(trackRoot,'trackData_ds7_full.mat'),         'ds7',         'ds7.bin',         'time_push_matched',  110, 150, 50)
  mkjob(fullfile(trackRoot,'trackData_ds8_full.mat'),         'ds8',         'ds8.bin',         'time_push_scer',     110, 150, 50)
};

if nargin >= 1 && ~isempty(varargin{1}), jobs = varargin{1}; else, jobs = defaultJobs; end
if nargin >= 2 && ~isempty(varargin{2}), dst  = varargin{2}; else, dst  = dstDefault; end

cols = {'scenario','source_file','spoof_type','prn','epoch_idx','t_sec', ...
        'segment','label','label_name','cn0_dbhz','mean_cn0_dbhz','noise_cn0', ...
        'doppler_hz','carr_freq_hz','i_prompt','q_prompt','prompt_power', ...
        'prompt_phase_rad','dll_discr','fll_filter','pll_lock','fll_lock'};
fmt = ['%s,%s,%s,%d,%d,%.4f,%s,%d,%s,', ...          % id + label (1-9)
       '%.4f,%.4f,%.6f,%.4f,%.4f,%.6g,%.6g,%.6g,', ... % cn0 family + doppler + prompt (10-17)
       '%.6f,%.6f,%.6f,%.4f,%.4f\n'];                 % phase + loops + locks (18-22)

dstDir = fileparts(dst);
if ~isempty(dstDir) && ~isfolder(dstDir), mkdir(dstDir); end
fid = fopen(dst, 'w');
if fid < 0, error('export_texbat_track:open', 'cannot open %s', dst); end
fprintf(fid, '%s\n', strjoin(cols, ','));

tot = struct('rows',0,'gen',0,'spoof',0,'trans',0,'skip',0);
for k = 1:size(jobs,1)
    s = export_one(fid, fmt, jobs{k,1});
    f = fieldnames(tot);
    for m = 1:numel(f), tot.(f{m}) = tot.(f{m}) + s.(f{m}); end
end
fclose(fid);
fprintf(['\nWrote %s\n  rows=%d  genuine=%d  spoof=%d  ', ...
         '(transition dropped=%d, untracked skipped=%d)\n'], ...
        dst, tot.rows, tot.gen, tot.spoof, tot.trans, tot.skip);
end

% =============================================================================
function j = mkjob(trackMat, scenario, sourceFile, spoofType, onsetLo, onsetHi, decim)
j = struct('trackMat',trackMat,'scenario',scenario,'sourceFile',sourceFile, ...
           'spoofType',spoofType,'onsetLo',onsetLo,'onsetHi',onsetHi,'decim',decim);
end

% =============================================================================
function s = export_one(fid, fmt, j)
s = struct('rows',0,'gen',0,'spoof',0,'trans',0,'skip',0);
if ~isfile(j.trackMat), fprintf('  SKIP missing %s\n', j.trackMat); return; end
D = load(j.trackMat);
if ~isfield(D, 'trackData'), fprintf('  SKIP no trackData in %s\n', j.trackMat); return; end
trackData = D.trackData;
if isfield(D, 'settings') && isfield(D.settings, 'sys')
    signal = D.settings.sys.enabledSignals{1};
else
    signal = 'gpsl1';
end
if ~isfield(trackData, signal)
    fprintf('  SKIP %s: no signal field %s\n', j.scenario, signal); return;
end
tg = trackData.(signal);

for c = 1:numel(tg.channel)
    ch = tg.channel(c);
    if ~isfield(ch, 'SvId') || ~isfield(ch.SvId, 'satId'), continue; end
    prn = ch.SvId.satId;
    if ~isfield(ch, 'CN0fromSNR'), continue; end
    n = numel(ch.CN0fromSNR);
    for i = 1:j.decim:n
        cn0 = at(ch, 'CN0fromSNR', i);
        if ~isfinite(cn0) || cn0 <= 0, s.skip = s.skip + 1; continue; end   % untracked epoch
        tSec = (i - 1) / 1000;                         % 1 ms tracking epochs
        if tSec < j.onsetLo
            lbl = 0; lname = 'genuine';     seg = 'pre_onset';
        elseif tSec >= j.onsetHi
            lbl = 1; lname = 'counterfeit'; seg = 'post_onset';
        else
            s.trans = s.trans + 1; continue;           % takeover transition, dropped
        end
        ip = at(ch, 'I_P', i);  qp = at(ch, 'Q_P', i);
        fprintf(fid, fmt, ...
            j.scenario, j.sourceFile, j.spoofType, prn, i, tSec, seg, lbl, lname, ...
            cn0, at(ch,'meanCN0fromSNR',i), at(ch,'noiseCNOfromSNR',i), ...
            at(ch,'doppler',i), at(ch,'carrFreq',i), ip, qp, ip^2 + qp^2, ...
            atan2(qp, ip + eps), at(ch,'dllDiscr',i), at(ch,'fllFilter',i), ...
            at(ch,'pllLockIndicator',i), at(ch,'fllLockIndicator',i));
        s.rows = s.rows + 1;
        if lbl, s.spoof = s.spoof + 1; else, s.gen = s.gen + 1; end
    end
    fprintf('  %-11s PRN %2d: %d epochs kept\n', j.scenario, prn, ...
            numel(1:j.decim:n));
end
end

% =============================================================================
function v = at(ch, name, i)
% One per-epoch value with bounds/existence guard (NaN if unavailable).
v = NaN;
if isfield(ch, name)
    x = ch.(name);
    if ~isempty(x) && i >= 1 && i <= numel(x), v = double(x(i)); end
end
end
