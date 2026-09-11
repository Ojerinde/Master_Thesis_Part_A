function info = make_spoof_composite(cfg)
% MAKE_SPOOF_COMPOSITE  Synthesize a GPS L1 C/A spoofing waveform and sum it into
% an authentic TEXBAT window, producing an IF sample file FGI-GSRx can track.
%
% The spoofer is seeded from the receiver's own tracking of the authentic
% recording, so the counterfeit signal is code- and carrier-aligned to the
% authentic one at takeover, in the manner of a phase-aligned TEXBAT takeover.
% The attacker's variables are transmitter settings, not feature perturbations:
% power advantage, code offset and pull-off rate, Doppler offset, carrier phase.
%
% Because the output is a sample file and the observables come from FGI-GSRx
% tracking it, every constraint the receiver imposes (carrier and code
% continuity, loop bandwidth and transient response, front-end saturation,
% epoch-to-epoch trajectory smoothness) holds by construction rather than by
% assertion.
%
% cfg fields
%   authFile   - authentic IF file (TEXBAT .bin, 25 Msps complex int16 I/Q)
%   trackMat   - trackData .mat from tracking authFile (seeds the spoofer)
%   outFile    - composite IF file to write
%   tStart     - window start, seconds from the start of authFile
%   tDur       - window duration, seconds
%   prns       - PRNs to spoof; [] spoofs every tracked channel
%   spoof      - struct of transmitter parameters (see defaults below)
%   selfTest   - true to run the software correlation check and skip writing
%   agc        - model the front-end AGC by rescaling the composite so its peak
%                matches the authentic peak, rather than letting it clip. A real
%                front end answers added power with gain reduction, so the cost
%                of a power advantage is quantisation loss on the authentic
%                signal, not clipping. Default true. Set false to measure what
%                clipping alone would do.
%
% cfg.spoof fields (scalar, or per-PRN vector matching cfg.prns)
%   alphaDb    - spoof-to-authentic power advantage, dB
%   tau0Chips  - code offset at takeover, chips
%   dtauChipS  - code pull-off rate, chips per second (positive = range grows)
%   dfdHz      - Doppler offset relative to the authentic carrier, Hz
%   phi0Rad    - carrier phase offset at takeover, radians
%   rampSec    - duration over which the power advantage rises from zero
%
% Returns an info struct carrying the realised parameters, the saturation
% count, and, for a self test, the per-PRN alignment correlation.

% ---------------------------------------------------------------- defaults
fgiRoot = 'D:\BEIHANG UNIVERSITY\Research\code\FGI-GSRx';
if isfolder(fgiRoot), addpath(genpath(fgiRoot)); end

cfg = withDefault(cfg, 'prns',     []);
cfg = withDefault(cfg, 'selfTest', false);
cfg = withDefault(cfg, 'signal',   'gpsl1');
cfg = withDefault(cfg, 'chunk',    2^22);
% The AGC gain is set from the composite peak. Measuring it over a prefix rather
% than the whole window halves the synthesis cost, and a prefix of a second at
% 25 Msps holds 25 million samples, which fixes the peak of a noise-like signal
% closely enough to set a gain.
cfg = withDefault(cfg, 'agcProbeSec', 1.0);
% Diagnostic: also correlate the composite against each replica, to tell a fault
% in the written samples apart from one downstream of them. Costs one buffer per
% channel, so use a small chunk with it.
cfg = withDefault(cfg, 'sumTest', false);
if ~isfield(cfg,'agc') || isempty(cfg.agc), cfg.agc = true; end
if cfg.sumTest, cfg.selfTest = true; cfg.chunk = min(cfg.chunk, 2^19); end

sp = withDefault(getfielddef(cfg,'spoof',struct()), 'alphaDb',   0.0);
sp = withDefault(sp, 'tau0Chips', 0.0);
sp = withDefault(sp, 'dtauChipS', 0.0);
sp = withDefault(sp, 'dfdHz',     0.0);
sp = withDefault(sp, 'phi0Rad',   0.0);
sp = withDefault(sp, 'rampSec',   0.0);
% Duration of the ds7 phasor takeover. When >0 it overrides alphaDb/rampSec and
% follows the published capture law; see the takeover block in the chunk loop.
sp = withDefault(sp, 'takeoverSec', 0.0);

% TEXBAT front end. centerFrequency equals carrierFreq, so the recording is at
% baseband and the stored carrFreq is the baseband carrier of each channel.
FS          = 25e6;      % sampling frequency, Hz
CODE_FREQ   = 1.023e6;   % chipping rate, Hz
CODE_CHIPS  = 1023;      % chips per code period
BYTES_PER   = 4;         % int16 I + int16 Q
ADC_MAX     = 32767;

n0 = round(cfg.tStart * FS) + 1;             % first sample, 1-based, file-relative
nN = round(cfg.tDur   * FS);                 % samples in the window
n1 = n0 + nN - 1;

% ---------------------------------------------------------------- seeding
D = load(cfg.trackMat, 'trackData');
if ~isfield(D, 'trackData'), error('make_spoof_composite:seed', ...
        'no trackData in %s', cfg.trackMat); end
tg = D.trackData.(cfg.signal);

chans = struct('prn',{},'sIdx',{},'chip0',{},'carr',{},'amp',{},'bit',{});
for c = 1:numel(tg.channel)
    ch = tg.channel(c);
    if ~isfield(ch,'SvId') || ~isfield(ch,'absoluteSample'), continue; end
    prn = double(ch.SvId.satId);
    if ~isempty(cfg.prns) && ~ismember(prn, cfg.prns), continue; end

    aS  = double(ch.absoluteSample(:));
    cn0 = double(ch.CN0fromSNR(:));
    ip  = double(ch.I_P(:));  qp = double(ch.Q_P(:));
    cf  = double(ch.carrFreq(:));

    % Keep epochs that are tracked and that bracket the requested window. One
    % epoch of margin each side so interpolation covers the whole window.
    ok  = isfinite(aS) & aS > 0 & isfinite(cn0) & cn0 > 0;
    k   = find(ok & aS >= n0 - 2*FS/1000 & aS <= n1 + 2*FS/1000);
    if numel(k) < 10
        fprintf('  PRN %2d: only %d usable epochs in window, skipped\n', prn, numel(k));
        continue;
    end

    % Samples per coherent integration, used to invert the correlation gain:
    % a signal of amplitude A aligned over Nc samples correlates to A*Nc.
    nc = median(diff(aS(k)));

    e.prn   = prn;
    e.sIdx  = aS(k);                      % epoch boundaries, absolute samples
    e.chip0 = (0:numel(k)-1)' * CODE_CHIPS;  % chips accumulated at each boundary
    e.carr  = cf(k);                      % baseband carrier frequency, Hz
    e.amp   = sqrt(ip(k).^2 + qp(k).^2) / nc;   % authentic amplitude, ADC units
    e.bit   = sign(ip(k)); e.bit(e.bit == 0) = 1;
    chans(end+1) = e; %#ok<AGROW>
end
if isempty(chans), error('make_spoof_composite:seed','no usable channels'); end

np  = numel(chans);
par = struct();
for f = {'alphaDb','tau0Chips','dtauChipS','dfdHz','phi0Rad','rampSec','takeoverSec'}
    v = sp.(f{1});
    if isscalar(v), v = repmat(v, 1, np); end
    if numel(v) ~= np, error('make_spoof_composite:par', ...
            '%s has %d entries for %d channels', f{1}, numel(v), np); end
    par.(f{1}) = v(:)';
end

codes = zeros(np, CODE_CHIPS);
for p = 1:np, codes(p,:) = gpsl1GeneratePrnCode(chans(p).prn); end
codes(codes == 0) = -1;                    % map to +/-1 if returned as 0/1

% Interpolants over the epoch grid, built once. Rebuilding them per chunk, as a
% bare interp1 call does, dominates the synthesis cost.
Fchip = cell(1,np); Fcarr = cell(1,np); Famp = cell(1,np); Fbit = cell(1,np);
for p = 1:np
    e = chans(p);
    Fchip{p} = griddedInterpolant(e.sIdx, e.chip0, 'linear', 'linear');
    Fcarr{p} = griddedInterpolant(e.sIdx, e.carr,  'linear', 'linear');
    Famp{p}  = griddedInterpolant(e.sIdx, e.amp,   'linear', 'linear');
    Fbit{p}  = griddedInterpolant(e.sIdx, e.bit,   'previous', 'nearest');
end

fidIn = fopen(cfg.authFile, 'rb');
if fidIn < 0, error('make_spoof_composite:open','cannot read %s', cfg.authFile); end
cleanIn = onCleanup(@() fclose(fidIn));

fidOut = -1;
if ~cfg.selfTest
    fidOut = fopen(cfg.outFile, 'wb');
    if fidOut < 0, error('make_spoof_composite:open','cannot write %s', cfg.outFile); end
    cleanOut = onCleanup(@() fclose(fidOut)); %#ok<NASGU>
end

info = struct('prns', [chans.prn], 'nSamples', nN, 'saturated', 0, ...
              'peakAuth', 0, 'peakRaw', 0, 'peakOut', 0, 'agcDb', 0, ...
              'corr', zeros(1,np), 'corrSum', zeros(1,np), ...
              'authPower', 0, 'spoofPower', 0, 'params', par);
S = cell(1, np);

% Pass 1 measures the composite peak so the AGC gain can be set; pass 2 writes.
% A single pass suffices for the self test and when the AGC is disabled.
gAgc  = 1;
nPass = 1 + (cfg.agc && ~cfg.selfTest);

for pass = 1:nPass
    writing = (~cfg.selfTest) && (pass == nPass);
    fseek(fidIn, (n0 - 1) * BYTES_PER, 'bof');

    % Carrier phase accumulated from the window start, per channel. Integrating
    % the tracked carrier frequency reproduces the authentic phase trajectory up
    % to a constant, and that constant is phi0Rad, an attacker parameter.
    phaseAcc = zeros(1, np);
    info.authPower = 0; info.spoofPower = 0; info.corr = zeros(1,np);
    info.corrSum = zeros(1,np);
    info.peakAuth = 0;  info.peakRaw = 0;

    % Pass 1 exists only to set the AGC gain. A short prefix is a valid probe only
    % when the spoofer holds constant power. With a takeover ramp the prefix sits
    % at a fraction of full power, so the AGC sets no backoff and the full-power
    % section then clips: a ramped run measured this way saturated 1.6% of its
    % samples. Probe the prefix when there is no ramp, the whole window when
    % there is one.
    nThis = nN;
    if cfg.agc && ~cfg.selfTest && pass == 1 && max(par.rampSec) <= 0
        nThis = min(nN, round(cfg.agcProbeSec * FS));
    end

    done = 0;
    while done < nThis
        m   = min(cfg.chunk, nThis - done);
        raw = fread(fidIn, [1, 2*m], 'int16');
        if numel(raw) ~= 2*m, error('make_spoof_composite:read', ...
                'short read at sample %d of %d', done, nN); end
        x = raw(1:2:end) + 1i * raw(2:2:end);

        nAbs = (n0 + done) + (0:m-1);      % absolute sample index of each sample
        t    = (done + (0:m-1)) / FS;      % seconds from window start
        s    = zeros(1, m);

        for p = 1:np
            % Code phase. Interpolating accumulated chips against the receiver's
            % own epoch boundaries carries the authentic code Doppler exactly.
            % ds7 drifts the code only AFTER the takeover completes (takeover 110-130 s,
            % drift from 150 s), so the pull-off clock starts at the end of takeover.
            chips = Fchip{p}(nAbs) + par.tau0Chips(p) ...
                    + par.dtauChipS(p) * max(0, t - par.takeoverSec(p));
            idx   = mod(floor(chips), CODE_CHIPS) + 1;
            cw    = codes(p, idx);

            % Carrier phase, integrated sample by sample so it is continuous
            % across chunk boundaries.
            fc  = Fcarr{p}(nAbs) + par.dfdHz(p);
            ph  = phaseAcc(p) + (2*pi/FS) * cumsum(fc);
            phaseAcc(p) = ph(end);

            % Navigation bits, held over each epoch, as a receiver-based spoofer
            % recovers them from the authentic signal.
            bw = Fbit{p}(nAbs);

            % Amplitude, with the takeover ramp.
            aA = Famp{p}(nAbs);
            % Amplitude and carrier phase of the counterfeit.
            if par.takeoverSec(p) > 0
                % TEXBAT ds7 takeover law (Humphreys, "TEXBAT Data Sets 7 and 8",
                % Section 4): the relative phasor angle rises from pi/2 to pi over
                % the takeover, and the amplitude follows
                %     A_s(t) = -2 A_a cos[theta(t)].
                % Those two together hold the COMBINED phasor at constant magnitude
                % A_a while its angle rotates from 0 to pi:
                %     A_a + A_s e^{j theta} = -A_a e^{j 2 theta}.
                % That is what walks a receiver's loops from the authentic signal
                % onto the counterfeit with no power or phase transient, so the
                % spoofer captures the code tracking. Holding the steady state
                % alone (a constant 2x antipodal phasor) does not capture a
                % receiver that is already locked to the authentic signal.
                th = (pi/2) * (1 + min(1, max(0, t / par.takeoverSec(p))));
                gv = -2 * cos(th);
                sp_p = (aA .* gv) .* bw .* cw ...
                       .* exp(1i * (ph + par.phi0Rad(p) + th));
            else
                g  = 10^(par.alphaDb(p)/20);
                if par.rampSec(p) > 0
                    g = g * min(1, max(0, t / par.rampSec(p)));
                end
                sp_p = (aA .* g) .* bw .* cw .* exp(1i * (ph + par.phi0Rad(p)));
            end
            s    = s + sp_p;

            if cfg.selfTest
                % Correlating the replica against the authentic samples peaks
                % only if the code and carrier alignment is right.
                info.corr(p) = info.corr(p) + sum(x .* conj(sp_p));
                if cfg.sumTest, S{p} = sp_p; end
            end
        end

        if cfg.selfTest && cfg.sumTest
            % Correlate the composite, not just the authentic samples, against
            % each replica. Coherent addition doubles the recovered amplitude;
            % incoherent addition does not. This separates a fault in the samples
            % written to disk from anything that happens downstream of them.
            for p = 1:np
                info.corrSum(p) = info.corrSum(p) + sum((x + s) .* conj(S{p}));
            end
        end

        info.authPower  = info.authPower  + sum(abs(x).^2);
        info.spoofPower = info.spoofPower + sum(abs(s).^2);
        info.peakAuth   = max(info.peakAuth, max(max(abs(real(x))), max(abs(imag(x)))));

        if ~cfg.selfTest
            y = x + s;
            info.peakRaw = max(info.peakRaw, max(max(abs(real(y))), max(abs(imag(y)))));
        end

        if writing
            % The AGC scales the whole composite, authentic and counterfeit
            % alike, so a power advantage costs quantisation headroom on the
            % authentic signal rather than clipping the sum.
            yi = max(-ADC_MAX, min(ADC_MAX, round(gAgc * real(y))));
            yq = max(-ADC_MAX, min(ADC_MAX, round(gAgc * imag(y))));
            info.saturated = info.saturated + sum(abs(gAgc*real(y)) > ADC_MAX) ...
                                            + sum(abs(gAgc*imag(y)) > ADC_MAX);
            info.peakOut = max(info.peakOut, max(max(abs(yi)), max(abs(yq))));
            out = zeros(1, 2*m);
            out(1:2:end) = yi; out(2:2:end) = yq;
            fwrite(fidOut, out, 'int16');
        end

        done = done + m;
    end

    if cfg.agc && ~cfg.selfTest && pass == 1
        % Hold the composite at the authentic peak level, which is what an AGC
        % settled on the authentic signal would do once the spoofer switches on.
        gAgc = min(1, info.peakAuth / max(info.peakRaw, 1));
        info.agcDb = 20*log10(gAgc);
    end
end

if cfg.selfTest
    % Correlating the replica against the authentic samples yields both the
    % alignment quality and the carrier phase error. The magnitude, normalised
    % by what a perfectly aligned replica would recover, is the code alignment.
    % The angle is the phase by which this channel's replica lags the authentic
    % carrier, and adding it to phi0Rad makes the counterfeit genuinely carrier
    % phase aligned rather than merely free of a frequency offset.
    info.phaseErr = zeros(1, np);
    for p = 1:np
        ref = sum(interp1(chans(p).sIdx, chans(p).amp, ...
                          n0:(n0+nN-1), 'linear', 'extrap').^2);
        info.phaseErr(p) = angle(info.corr(p));
        info.corr(p)     = abs(info.corr(p)) / max(ref, eps);
        info.corrSum(p)  = abs(info.corrSum(p)) / max(ref, eps);
    end
end
end

% =============================================================================
function s = withDefault(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end

function v = getfielddef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
