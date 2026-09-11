function out = validate_against_ds7(varargin)
% VALIDATE_AGAINST_DS7  Reproduce TEXBAT ds7 from its published specification and
% compare the tracked observables against the real recording.
%
% This is the gate that decides whether the synthesizer may be trusted for
% spoofer configurations TEXBAT does not contain. A generator that cannot
% reproduce a documented scenario is not evidence about undocumented ones.
%
% Specification source
%   T. E. Humphreys, "TEXBAT Data Sets 7 and 8", University of Texas at Austin
%   Radionavigation Laboratory, 16 March 2016, Section 4 (local copy:
%   data/raw/texbat/texbat_complementary files/texbat_ds7_and_ds8.pdf).
%
% What Section 4 specifies, and how each item maps to a transmitter parameter:
%
%   Interval 110-130 s. The authentic phasor is P_a = A_a, taken real. The
%   injected spoofing phasor is P_s(t) = A_s(t) exp[j theta(t)] with theta rising
%   linearly from pi/2 to pi and A_s(t) = -2 A_a cos[theta(t)]. Those two choices
%   together give
%       P_as = A_a + A_s e^{j theta} = -A_a e^{j 2 theta},
%   a combined signal of constant magnitude A_a whose phase rotates from 0 to pi.
%   That is what lets the spoofer take the victim's loops without a power or
%   phase transient.
%
%   Interval 130-150 s. Held there. The spoofing phasor is -2 A_a: twice the
%   authentic amplitude, antipodally aligned. Authentic plus spoof is therefore
%   A_a - 2 A_a = -A_a, so the victim's C/N0 and pseudorange match cleanStatic
%   and only the carrier phase differs, by pi. This is why the real recording
%   sits at roughly clean C/N0 rather than above it, and it is the item that a
%   matched-power in-phase imitation gets wrong.
%       -> alphaDb = 20*log10(2) = 6.0206, phi0Rad = pi
%
%   Interval 150-400 s. The relative code phase of every spoofing signal grows
%   from zero at 1.2 metres per second, inducing a clock offset in the victim,
%   while "the Doppler frequency of the spoofing signals remains exactly as the
%   Doppler frequency of the authentic signals", which the document names a
%   frequency-locked attack.
%       -> dtauChipS = 1.2 / (c/f_chip) chips per second, dfdHz = 0
%
%   The document's own closing check: by t = 468 s the accumulated relative code
%   phase is 381.6 m, a 1.273 us clock offset. That is 1.2 m/s over the 318 s
%   from 150 s, and 381.6/c = 1.273 us, so the drift rate and its start are
%   mutually consistent.
%
% The window defaults to 150-160 s, the first ten seconds of the drift interval,
% where the code offset spans 0 to 12 m and the two signals still overlap within
% a chip.

C_LIGHT = 299792458;
F_CHIP  = 1.023e6;
CHIP_M  = C_LIGHT / F_CHIP;          % 293.0523 m per chip

p = inputParser;
addParameter(p, 'tStart',  150);     % start of the ds7 code-drift interval
addParameter(p, 'tDur',    10);
addParameter(p, 'driftMps', 1.2);    % Section 4, interval 150-400 s
addParameter(p, 'workDir', 'D:\BEIHANG UNIVERSITY\Research\data\FGI_Data\composite');
addParameter(p, 'name',    'ds7spec');
parse(p, varargin{:});
a = p.Results;

spoof = struct( ...
    'alphaDb',   20*log10(2), ...    % spoof at twice the authentic amplitude
    'phi0Rad',   pi, ...             % antipodally aligned with the authentic
    'dfdHz',     0, ...              % frequency-locked to the authentic Doppler
    'tau0Chips', 0, ...              % drift starts from zero relative code phase
    'dtauChipS', a.driftMps / CHIP_M, ...
    'rampSec',   0);                 % takeover already complete at t = 150 s

fprintf(['ds7 from specification: alpha %+.4f dB, phi0 %.4f rad, dfd %.1f Hz, ' ...
         'drift %.4f chip/s (%.2f m/s)\n'], spoof.alphaDb, spoof.phi0Rad, ...
        spoof.dfdHz, spoof.dtauChipS, a.driftMps);

out = run_composite_track(a.name, spoof, 'tStart', a.tStart, 'tDur', a.tDur, ...
                          'workDir', a.workDir);
end
