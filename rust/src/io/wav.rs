use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WavError {
    #[error("Failed to read WAV file: {0}")]
    ReadError(#[from] hound::Error),
    #[error("Unsupported sample format")]
    UnsupportedFormat,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Read a WAV file and return samples as [channels][samples]
pub fn read_wav(path: &Path) -> Result<(Vec<Vec<f32>>, u32), WavError> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
    };

    let num_samples = samples.len() / channels;
    let mut output = vec![vec![0.0f32; num_samples]; channels];

    for (i, sample) in samples.iter().enumerate() {
        let channel = i % channels;
        let sample_idx = i / channels;
        output[channel][sample_idx] = *sample;
    }

    Ok((output, sample_rate))
}

/// Write samples to a WAV file
/// Input: [channels][samples]
pub fn write_wav(path: &Path, samples: &[Vec<f32>], sample_rate: u32) -> Result<(), WavError> {
    let channels = samples.len();
    if channels == 0 {
        return Ok(());
    }

    let num_samples = samples[0].len();

    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for i in 0..num_samples {
        for ch in 0..channels {
            writer.write_sample(samples[ch][i])?;
        }
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;


    #[test]
    fn test_wav_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_roundtrip.wav");

        let original = vec![
            vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5],
            vec![0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5],
        ];

        write_wav(&test_path, &original, 44100).unwrap();

        let (loaded, sample_rate) = read_wav(&test_path).unwrap();

        assert_eq!(sample_rate, 44100);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].len(), 8);

        for ch in 0..2 {
            for i in 0..8 {
                assert!(
                    (loaded[ch][i] - original[ch][i]).abs() < 1e-5,
                    "Mismatch at ch={}, i={}: {} vs {}",
                    ch,
                    i,
                    loaded[ch][i],
                    original[ch][i]
                );
            }
        }

        fs::remove_file(&test_path).ok();
    }
}
