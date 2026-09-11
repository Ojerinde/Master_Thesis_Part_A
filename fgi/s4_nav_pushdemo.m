function s4_nav_pushdemo(varargin)
% S4  Navigation-stage confirmation: an attack that evades the detector also
% moves the receiver's solution.
%
% The library attacks apply a uniform code pull-off across every satellite. A
% static receiver absorbs a common-mode range ramp into its clock, so the
% induced error is a receiver time error, the TEXBAT ds7 attack class (ds7's
% documented effect is a 1.273 us clock offset). This driver synthesizes one
% RandomForest-evading config over a long window, tracks it, runs the nav stage,
% and exports the clock and position solution over time.
%
% No separate authentic baseline is run. The receiver is static, so its clock
% and position with no spoof are flat by construction; the attack's ramp is the
% effect. The window is 90 s so the frame decoder can recover an ephemeris (a
% 30 s window is too short) and the drift is large enough to see.
%
% Usage
%   s4_nav_pushdemo()                 % +3 dB, 14.65 m/s, 90 s window from 100 s
%   s4_nav_pushdemo(alphaDb, chipRate, tStart, tDur)

fgiRoot   = 'D:\BEIHANG UNIVERSITY\Research\code\FGI-GSRx';
scriptDir = 'D:\BEIHANG UNIVERSITY\Research\code\scripts';
addpath(genpath(fgiRoot)); addpath(scriptDir);
workDir = 'D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\composite';
outDir  = 'D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\s4_nav';
if ~isfolder(outDir), mkdir(outDir); end

CHIP_M = 299792458/1.023e6;         % 293.05 m per chip
alphaDb  = getdef(varargin, 1, 3.0);       % +3 dB captures the receiver
chipRate = getdef(varargin, 2, 0.05);      % 0.05 chip/s = 14.65 m/s pull-off
tStart   = getdef(varargin, 3, 100);
tDur     = getdef(varargin, 4, 90);
name = sprintf('s4_a%g_%gmps', alphaDb, round(chipRate*CHIP_M));

% Code-carrier consistent Doppler for this pull-off (align_frac = 1, no beat).
spoof = struct('alphaDb',alphaDb,'tau0Chips',0,'dtauChipS',chipRate, ...
               'dfdHz',-1540*chipRate,'phi0Rad',0,'rampSec',0);
fprintf(['S4 %s: +%.1f dB, pull-off %.1f m/s (%.4f chip/s), window %g-%g s\n' ...
         'predicted clock ramp %.1f ns/s -> %.2f us over %g s\n'], ...
        name, alphaDb, chipRate*CHIP_M, chipRate, tStart, tStart+tDur, ...
        chipRate*CHIP_M/299792458*1e9, chipRate*CHIP_M/299792458*tDur*1e6, tDur);

% Synthesize + track, keep the tracked .mat for the nav stage, nav at 1 Hz.
run_composite_track(name, spoof, 'tStart',tStart, 'tDur',tDur, ...
    'workDir',workDir, 'keepMat',true, 'warmupSec',1.5, 'navSolPeriod',1000);

attMat = fullfile(workDir, ['trackData_' name '.mat']);
attNav = fullfile(outDir,  ['navData_' name '.mat']);
fprintf('[S4] nav stage on the attack composite...\n');
run_fgi_nav(attMat, attNav);
export_fgi_nav({struct('navMat',attNav,'scenario',name,'sourceFile',[name '.bin'], ...
                'spoofType','attack','onsetLo',-inf,'onsetHi',-inf,'decim',1)}, ...
               fullfile(outDir, [name '_nav.csv']));
fprintf('[S4] done -> %s\n', fullfile(outDir, [name '_nav.csv']));
end

function v = getdef(args, i, d)
if numel(args) >= i && ~isempty(args{i}), v = args{i}; else, v = d; end
end
