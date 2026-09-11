function R = calibrate_spoof_alignment(varargin)
% CALIBRATE_SPOOF_ALIGNMENT  Gate S1 for the waveform-level attack.
%
% Sweeps the code offset of the synthesized replica against the authentic TEXBAT
% samples and reports the correlation recovered per PRN. If the code-phase model
% in MAKE_SPOOF_COMPOSITE is right, each channel peaks at one common offset with
% a normalised correlation near one, and the correlation angle at that peak is
% the carrier phase that aligns the spoof to the authentic signal.
%
% A flat or peakless sweep means the seeding is wrong and nothing downstream can
% be trusted. Nothing here calls FGI-GSRx, so the gate runs in minutes.
%
% Usage
%   R = calibrate_spoof_alignment()
%   R = calibrate_spoof_alignment('tStart', 60, 'tDur', 0.5, 'taus', -1:0.1:1)

p = inputParser;
addParameter(p, 'authFile', ['D:\BEIHANG UNIVERSITY\Research\code\' ...
    'gnss_adversarial_research\data\raw\texbat\cleanStatic.bin']);
addParameter(p, 'trackMat', ['D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\' ...
    'out\trackData_cleanStatic_full.mat']);
addParameter(p, 'tStart', 60);
addParameter(p, 'tDur',   0.5);
addParameter(p, 'taus',   -1.5:0.125:1.5);
addParameter(p, 'prns',   []);
parse(p, varargin{:});
a = p.Results;

fprintf('Alignment sweep: t=[%g,%g) s, %d code offsets\n', ...
        a.tStart, a.tStart + a.tDur, numel(a.taus));

M = []; prns = [];
for k = 1:numel(a.taus)
    cfg = struct('authFile', a.authFile, 'trackMat', a.trackMat, ...
                 'tStart', a.tStart, 'tDur', a.tDur, 'prns', a.prns, ...
                 'selfTest', true, ...
                 'spoof', struct('alphaDb', 0, 'tau0Chips', a.taus(k)));
    info = make_spoof_composite(cfg);
    if isempty(M), prns = info.prns; M = zeros(numel(a.taus), numel(prns)); end
    M(k,:) = info.corr; %#ok<AGROW>
    fprintf('  tau = %+6.3f chips   mean corr %.3f   max %.3f\n', ...
            a.taus(k), mean(info.corr), max(info.corr));
end

[peakVal, peakIdx] = max(M, [], 1);
R = struct('taus', a.taus, 'prns', prns, 'corr', M, ...
           'peakTau', a.taus(peakIdx), 'peakCorr', peakVal, ...
           'authPeakAdc', info.peakAuth);

fprintf('\nPer-PRN alignment peak\n');
fprintf('  PRN   peak tau (chips)   peak corr\n');
for j = 1:numel(prns)
    fprintf('  %3d   %+8.3f          %8.3f\n', prns(j), R.peakTau(j), R.peakCorr(j));
end
fprintf('\nAuthentic peak sample magnitude: %d of 32767 ADC counts\n', info.peakAuth);
fprintf('Headroom before saturation: %.1f dB\n', 20*log10(32767/max(info.peakAuth,1)));

spread = max(R.peakTau) - min(R.peakTau);
fprintf('\nGate: peak offsets agree to within %.3f chips across PRNs\n', spread);
if spread <= 0.25 && median(R.peakCorr) >= 0.7
    fprintf('Gate PASSED: seeding reproduces the authentic code and carrier.\n');
else
    fprintf(['Gate FAILED: alignment is not consistent across channels. ' ...
             'Check the absoluteSample convention before proceeding.\n']);
end
end
