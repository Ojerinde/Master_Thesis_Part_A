function sweep_realizable_library(varargin)
% SWEEP_REALIZABLE_LIBRARY  Build the attacker's realizable feasible set.
%
% Each grid point is a spoofer transmitter configuration. For each one this
% synthesizes the counterfeit waveform, sums it into the authentic TEXBAT
% window, tracks the composite with FGI-GSRx, and appends the resulting
% observables to a library CSV tagged with the parameters that produced them.
%
% The library is the feasible set stated as data. An attack searching it is
% searching over signals a transmitter can emit, so no projection or validity
% enforcement is needed downstream.
%
% The sweep is resumable. Points already present in the manifest are skipped,
% so the run can be stopped and restarted.
%
% Grid axes
%   alphaDb    power advantage over the authentic signal. The 1.3 dB and 10 dB
%              points match TEXBAT ds3 and ds2, so the library brackets the
%              documented scenarios.
%   dtauChipS  code pull-off rate. One chip is 293.05 m, so the rate in chips
%              per second is a range rate in units of 293.05 m/s.
%   alignFrac  how much of the required carrier Doppler the spoofer withholds in
%              order to stay aligned with the authentic carrier. A spoofer
%              pulling the code at dtauChipS must shift the carrier by
%              -1540*dtauChipS Hz to remain code-carrier consistent, so
%              dfd = -(1 - alignFrac) * 1540 * dtauChipS.
%
%              This is the attacker's central dilemma and it has no free
%              corner. At alignFrac = 0 the spoofer is code-carrier consistent
%              but its carrier sits at a frequency offset from the authentic
%              one, and the two beat against each other at |dfd| Hz. Measured
%              on a matched-power ds7 imitation, that beat swings C/N0 between
%              20 and 57 dB-Hz and inflates the observable spread several fold,
%              which is loud. At alignFrac = 1 the spoofer holds carrier phase
%              and leaves no beat, which is what ds7 is documented to do, but
%              it must then let the code drift against a stationary carrier and
%              accept full code-carrier divergence.
%
%              A feature-space attack never meets this trade-off, because there
%              C/N0 and Doppler are independent coordinates that can both be set
%              to whatever evades the detector.
%   rampSec    duration over which the power advantage rises from zero.
%
%   ampPhase   the (power advantage, carrier phase) pairs to try. Carrier phase
%              cannot be left fixed, because it decides whether a given power
%              advantage is loud or silent.
%
%              Writing g for the spoof-to-authentic amplitude ratio and phi for
%              its carrier phase relative to the authentic signal, the combined
%              amplitude is |1 + g exp(j phi)|. That equals 1, leaving the
%              victim's received power unchanged, exactly on the locus
%
%                  g = -2 cos(phi),        phi in (90, 180] degrees.
%
%              This is not a construction of ours. It is the takeover law
%              TEXBAT ds7 is built on: Humphreys, TEXBAT Data Sets 7 and 8,
%              Section 4, specifies A_s(t) = -2 A_a cos[theta(t)] with theta
%              running from pi/2 to pi, which is this locus traversed from
%              g = 0 to g = 2, and which is why the document can state that the
%              victim's C/N0 and pseudorange are indistinguishable from clean
%              while the spoofer takes the loops.
%
%              A spoofer at twice the authentic amplitude is +9.5 dB and
%              obvious if it adds in phase, and invisible in the power domain if
%              it adds antipodally. A grid that fixes phi at zero and sweeps
%              power alone would therefore miss the entire region the only
%              documented subtle attack occupies. The grid carries both the
%              in-phase family and the amplitude-neutral locus.

% Constants derived rather than tabulated, so they cannot drift from the receiver
% the composites are tracked with.
%   c        WGS-84 / IS-GPS-200 defined speed of light, exact by definition
%   F_L1     GPS L1 carrier, IS-GPS-200
%   F_CHIP   C/A chipping rate, IS-GPS-200
% One chip spans c/F_CHIP = 293.05 m of range.
% A code rate offset of one chip per second is a range rate of that many metres
% per second, which the carrier must answer with a Doppler of F_L1/F_CHIP = 1540
% Hz. That ratio is exact for GPS L1 C/A (1575.42 = 1540 x 1.023 MHz) and is the
% same quantity FGI-GSRx forms as carrToCodeRatio in allocateTrackChannel.m.
C_LIGHT  = 299792458;
F_L1     = 1575.42e6;
F_CHIP   = 1.023e6;
CHIP_M       = C_LIGHT / F_CHIP;   % 293.0523 m per chip
HZ_PER_CHIPS = F_L1 / F_CHIP;      % 1540 Hz of carrier per chip/s of code

p = inputParser;
addParameter(p, 'outDir',  'D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\composite');
addParameter(p, 'tStart',  60);
addParameter(p, 'tDur',    10);
% The 1.3 dB and 10 dB power points are ds3 and ds2. The 0.0041 chip/s pull-off
% is the 1.2 m/s code drift documented for ds7, the stealthiest published static
% attack, so the grid reaches below it rather than starting above it.
addParameter(p, 'alphaDb',    [-3 0 1.3 3 6 10]);   % in-phase family, phi = 0
addParameter(p, 'neutralDeg', [100 120 140 160 180]); % amplitude-neutral locus
addParameter(p, 'dtauChipS',  [0 0.002 0.0041 0.01 0.05 0.2 0.5]);
addParameter(p, 'alignFrac',  [0 0.5 1.0]);
% The library window sits after takeover, so the ramp is not exercised within it.
% Takeover dynamics are a separate question from steady-state detectability.
addParameter(p, 'rampSec',    0);
addParameter(p, 'dryRun',  false);
parse(p, varargin{:});
a = p.Results;

if ~isfolder(a.outDir), mkdir(a.outDir); end
libCsv   = fullfile(a.outDir, 'realizable_library.csv');
manCsv   = fullfile(a.outDir, 'realizable_manifest.csv');

% Amplitude and carrier phase travel together. The in-phase family is the naive
% spoofer; the neutral locus g = -2 cos(phi) is the one that leaves the victim's
% received power unchanged, and is what ds7 uses.
ap = struct('alphaDb', {}, 'phiDeg', {}, 'family', {});
for k = 1:numel(a.alphaDb)
    ap(end+1) = struct('alphaDb', a.alphaDb(k), 'phiDeg', 0, ...
                       'family', "inphase"); %#ok<AGROW>
end
for k = 1:numel(a.neutralDeg)
    phi = a.neutralDeg(k);
    gAmp = -2 * cosd(phi);
    if gAmp <= 0, continue; end   % locus is only defined for phi > 90 degrees
    ap(end+1) = struct('alphaDb', 20*log10(gAmp), 'phiDeg', phi, ...
                       'family', "neutral"); %#ok<AGROW>
end

grid = [];
for i1 = 1:numel(ap)
 for i2 = 1:numel(a.dtauChipS)
  for i3 = 1:numel(a.alignFrac)
   for i4 = 1:numel(a.rampSec)
     g.alphaDb   = ap(i1).alphaDb;
     g.phiDeg    = ap(i1).phiDeg;
     g.family    = ap(i1).family;
     g.dtauChipS = a.dtauChipS(i2);
     g.alignFrac = a.alignFrac(i3);
     g.rampSec   = a.rampSec(i4);
     % A spoofer that is not pulling the code needs no Doppler offset, so every
     % alignFrac collapses to the same signal there. Keep one.
     if g.dtauChipS == 0 && g.alignFrac ~= 0, continue; end
     grid = [grid; g]; %#ok<AGROW>
   end
  end
 end
end

fprintf('Realizable library sweep: %d grid points, %.0f s window each\n', ...
        numel(grid), a.tDur);
fprintf('Estimated tracking cost: %.1f hours at 38x real time\n', ...
        numel(grid) * a.tDur * 38 / 3600);
if a.dryRun
    for k = 1:numel(grid)
        gk = grid(k);
        ca = abs(1 + 10^(gk.alphaDb/20) * exp(1i*deg2rad(gk.phiDeg)));
        fprintf(['  %3d  %-8s alpha %+6.2f dB  phi %3d deg  received %+5.2f dB  ' ...
                 'pulloff %6.1f m/s  align %4.2f  beat %6.1f Hz\n'], ...
                k, gk.family, gk.alphaDb, gk.phiDeg, 20*log10(ca), ...
                gk.dtauChipS*CHIP_M, gk.alignFrac, ...
                (1-gk.alignFrac)*HZ_PER_CHIPS*gk.dtauChipS);
    end
    return;
end

done = containers.Map('KeyType','char','ValueType','logical');
if isfile(manCsv)
    M = readtable(manCsv, 'TextType','string');
    for k = 1:height(M), done(char(M.point(k))) = true; end
    fprintf('Resuming: %d points already in the manifest\n', done.Count);
end

for k = 1:numel(grid)
    g    = grid(k);
    name = sprintf('a%+06.2f_p%03d_d%07.4f_g%04.2f', ...
                   g.alphaDb, g.phiDeg, g.dtauChipS, g.alignFrac);
    name = strrep(strrep(name, '+', 'p'), '-', 'm');
    name = strrep(name, '.', '_');
    if isKey(done, name)
        fprintf('[%3d/%d] %s  already done\n', k, numel(grid), name);
        continue;
    end

    % Carrier Doppler, reduced from the code-carrier consistent value by however
    % much of it the spoofer withholds to stay aligned with the authentic carrier.
    dfd = -(1 - g.alignFrac) * HZ_PER_CHIPS * g.dtauChipS;
    % phi0Rad is measured from the authentic carrier; run_composite_track adds the
    % per-channel offset that makes zero mean genuinely in phase.
    sp  = struct('alphaDb', g.alphaDb, 'tau0Chips', 0, ...
                 'dtauChipS', g.dtauChipS, 'dfdHz', dfd, ...
                 'phi0Rad', deg2rad(g.phiDeg), 'rampSec', g.rampSec);

    fprintf(['[%3d/%d] %s  %s  alpha %+.2f dB  phi %d deg  pulloff %.1f m/s  ' ...
             'align %.2f  beat %.1f Hz\n'], k, numel(grid), name, g.family, ...
            g.alphaDb, g.phiDeg, g.dtauChipS*CHIP_M, g.alignFrac, abs(dfd));
    try
        o = run_composite_track(name, sp, 'tStart', a.tStart, 'tDur', a.tDur, ...
                                'workDir', a.outDir);
    catch ME
        fprintf('  FAILED: %s\n', ME.message);
        continue;
    end

    T = readtable(o.csv);
    if isempty(T), fprintf('  empty export, skipped\n'); continue; end
    % Combined amplitude the victim receives, relative to authentic. Unity on the
    % neutral locus by construction, which is what makes that family quiet.
    gAmp    = 10^(g.alphaDb/20);
    combAmp = abs(1 + gAmp * exp(1i*deg2rad(g.phiDeg)));

    T.point       = repmat(string(name), height(T), 1);
    T.family      = repmat(g.family,     height(T), 1);
    T.alpha_db    = repmat(g.alphaDb,    height(T), 1);
    T.phi_deg     = repmat(g.phiDeg,     height(T), 1);
    T.comb_amp_db = repmat(20*log10(combAmp), height(T), 1);
    T.pulloff_mps = repmat(g.dtauChipS*CHIP_M, height(T), 1);
    T.align_frac  = repmat(g.alignFrac,  height(T), 1);
    T.beat_hz     = repmat(abs(dfd),     height(T), 1);
    writeTableAppend(T, libCsv);

    row = table(string(name), g.family, g.alphaDb, g.phiDeg, ...
                20*log10(combAmp), g.dtauChipS, g.dtauChipS*CHIP_M, ...
                dfd, g.alignFrac, abs(dfd), height(T), ...
                o.syn.agcDb, o.syn.saturated, o.tTrack, ...
        'VariableNames', {'point','family','alpha_db','phi_deg','comb_amp_db', ...
                          'dtau_chips_s','pulloff_mps','dfd_hz','align_frac', ...
                          'beat_hz','n_rows','agc_db','saturated','track_sec'});
    writeTableAppend(row, manCsv);
    delete(o.csv);
    fprintf('  %d rows, agc %+.2f dB, %.0f s\n', height(T), o.syn.agcDb, o.tTrack);
end

fprintf('\nLibrary written to %s\n', libCsv);
end

% =============================================================================
function writeTableAppend(T, path)
if isfile(path)
    writetable(T, path, 'WriteMode', 'append', 'WriteVariableNames', false);
else
    writetable(T, path);
end
end
