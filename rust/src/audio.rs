use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;
use anyhow::{Result, Context};

pub fn load_audio(path: &str) -> Result<(Vec<f32>, usize, u32)> {
    let src = File::open(path).context("failed to open audio file")?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        }
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("unsupported format")?;
    let mut format = probed.format;
    let track = format.default_track().context("no track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);

    let mut samples: Vec<f32> = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id { continue; }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sample_buf.samples());
            }
            Err(e) => {
                eprintln!("decode error: {}", e);
                break;
            }
        }
    }

    Ok((samples, channels, sample_rate))
}

pub fn save_audio(path: &str, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    let spec = hound::WavSpec {
        channels: channels,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    Ok(())
}
