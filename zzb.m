function [out_zzb] = zzb(snr_ratio_scalar, frequencyBins, signalPowerSpectrum, Tobs)
% This function calculates the Ziv-zakai bound of the sampled signal
% snr_ratio_scalar: Signal-to-noise ratio (NOTE: it is a scalar value, not in dB)
% Tobs: observation interval which defines the integration boundaries of the integral in ZZB (usually, it is set to signalPeriod/2)
% frequencyBins: vector containing values of all the pilot's frequencies (i.e. 256 for f. band 1 + 256 for f. band 2)
% signalPowerSpectrum: vector of power values allocated to each pilot (i.e. 256 for f. band 1 + 256 for f. band 2)
% NOTE: index 1 of frequencyBins corresponds to index 1 of specSig
fun      =@(x) x.* (1-x/Tobs) .* ( 0.5*erfc((1/sqrt(2))*sqrt(snr_ratio_scalar * (1-real(acf(x, frequencyBins, signalPowerSpectrum)) ) )) ); % function to be integrated for ZZB
out_zzb  = integral(fun, 0, Tobs, 'ArrayValued', true, 'AbsTol', 1e-14, 'RelTol', 0);

end

