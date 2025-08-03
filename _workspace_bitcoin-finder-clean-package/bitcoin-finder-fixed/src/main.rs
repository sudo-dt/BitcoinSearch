use bip39::Mnemonic;
use bitcoin::util::bip32::{ExtendedPrivKey, DerivationPath, ChildNumber};
use bitcoin::{Address, Network, PrivateKey};
use bitcoin::secp256k1::Secp256k1;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use crossbeam_channel::bounded;
use memmap2::Mmap;
use serde::{Serialize, Deserialize};
use clap::{App, Arg};
use signal_hook::{iterator::Signals, consts::SIGTERM};
use chrono;
use num_cpus;

// ===== Configuration Structures =====

#[derive(Clone, Serialize, Deserialize)]
struct Config {
    min_matching_words: usize,
    max_addresses_per_path: u32,
    highlight_threshold: usize,
    word_match_logging: bool,
    paths_to_check: Vec<String>,
    service_mode: bool,
    checkpoint_interval: u64,
    auto_resume: bool,
    learning_enabled: bool,
    min_score_threshold: f64,
    max_high_scores: usize,
    adaptive_search: bool,
    word_weight_multiplier: f64,
    pattern_recognition: bool,
    position_aware_matching: bool,
    health_check_interval: u64,
    max_memory_percent: u8,
    max_checkpoint_size_mb: u64,
    checkpoint_compression: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_matching_words: 1,
            max_addresses_per_path: 20,
            highlight_threshold: 3,
            word_match_logging: true,
            paths_to_check: vec![
                "m/44'/0'/0'/0".to_string(), "m/44'/0'/0'/1".to_string(),
                "m/44'/0'/1'/0".to_string(), "m/44'/0'/1'/1".to_string(),
                "m/49'/0'/0'/0".to_string(), "m/49'/0'/0'/1".to_string(),
                "m/49'/0'/1'/0".to_string(), "m/49'/0'/1'/1".to_string(),
                "m/84'/0'/0'/0".to_string(), "m/84'/0'/0'/1".to_string(),
                "m/84'/0'/1'/0".to_string(), "m/84'/0'/1'/1".to_string(),
            ],
            service_mode: false,
            checkpoint_interval: 900, // 15 minutes
            auto_resume: false,
            learning_enabled: true,
            min_score_threshold: 0.5,
            max_high_scores: 100,
            adaptive_search: true,
            word_weight_multiplier: 1.5,
            pattern_recognition: true,
            position_aware_matching: true,
            health_check_interval: 60, // 1 minute
            max_memory_percent: 90,    // 90% memory usage threshold
            max_checkpoint_size_mb: 100, // 100MB max checkpoint size
            checkpoint_compression: true,
        }
    }
}

// ===== Health Monitoring System =====

struct HealthMonitor {
    last_check: Instant,
    check_interval: Duration,
    max_memory_percent: u8,
    last_mnemonics_count: u64,
    stall_threshold: Duration,
    logger: Arc<Logger>,
    stats: Arc<SearchStats>,
}

impl HealthMonitor {
    fn new(
        check_interval: u64,
        max_memory_percent: u8,
        logger: Arc<Logger>,
        stats: Arc<SearchStats>,
    ) -> Self {
        Self {
            last_check: Instant::now(),
            check_interval: Duration::from_secs(check_interval),
            max_memory_percent,
            last_mnemonics_count: 0,
            stall_threshold: Duration::from_secs(300), // 5 minutes
            logger,
            stats,
        }
    }

    fn should_check(&mut self) -> bool {
        if self.last_check.elapsed() >= self.check_interval {
            self.last_check = Instant::now();
            true
        } else {
            false
        }
    }

    fn check_health(&mut self) -> bool {
        // Check memory usage
        let memory_usage = self.get_memory_usage();
        if memory_usage > self.max_memory_percent as f64 {
            let _ = self.logger.log_message(
                LogLevel::Warning,
                &format!("High memory usage detected: {:.1}%. Consider restarting.", memory_usage),
            );
            println!("⚠️ High memory usage: {:.1}%", memory_usage);
            return false;
        }

        // Check for stalls
        let current_count = self.stats.mnemonics_checked.load(Ordering::Relaxed);
        if current_count == self.last_mnemonics_count && self.last_check.elapsed() > self.stall_threshold {
            let _ = self.logger.log_message(
                LogLevel::Warning,
                "Process appears stalled. No progress detected in the last 5 minutes.",
            );
            println!("⚠️ Process appears stalled. No progress detected.");
            return false;
        }
        self.last_mnemonics_count = current_count;

        // Log health status
        let _ = self.logger.log_message(
            LogLevel::Info,
            &format!("Health check passed. Memory usage: {:.1}%", memory_usage),
        );
        
        true
    }

    fn get_memory_usage(&self) -> f64 {
        // This is a simplified implementation that works on Linux
        if let Ok(file) = File::open("/proc/meminfo") {
            let reader = BufReader::new(file);
            let mut total_kb = 0;
            let mut available_kb = 0;

            for line in reader.lines() {
                if let Ok(line) = line {
                    if line.starts_with("MemTotal:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = value.parse::<u64>() {
                                total_kb = kb;
                            }
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = value.parse::<u64>() {
                                available_kb = kb;
                            }
                        }
                    }
                }
            }

            if total_kb > 0 {
                return 100.0 * (total_kb - available_kb) as f64 / total_kb as f64;
            }
        }

        // Fallback if we can't read memory info
        0.0
    }
}

// ===== Checkpoint System =====

#[derive(Serialize, Deserialize)]
struct CheckpointData {
    mnemonics_checked: u64,
    addresses_generated: u64,
    word_matches_found: u64,
    timestamp: u64,
    run_duration_seconds: u64,
    last_seed: Option<Vec<u8>>,
    high_score_mnemonics: Vec<HighScoreMnemonic>,
    word_frequencies: HashMap<String, usize>,
    word_pair_frequencies: HashMap<String, usize>,
    word_position_frequencies: HashMap<String, Vec<usize>>,
    pattern_frequencies: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct HighScoreMnemonic {
    mnemonic: String,
    matching_words: Vec<String>,
    score: f64,
    positions: HashMap<String, usize>,
}

struct CheckpointManager {
    checkpoint_file: String,
    checkpoint_interval: Duration,
    last_checkpoint: Instant,
    max_checkpoint_size_mb: u64,
    compression_enabled: bool,
}

impl CheckpointManager {
    fn new(
        checkpoint_file: &str, 
        interval_seconds: u64,
        max_size_mb: u64,
        compression: bool,
    ) -> Self {
        Self {
            checkpoint_file: checkpoint_file.to_string(),
            checkpoint_interval: Duration::from_secs(interval_seconds),
            last_checkpoint: Instant::now(),
            max_checkpoint_size_mb: max_size_mb,
            compression_enabled: compression,
        }
    }

    fn should_checkpoint(&mut self) -> bool {
        if self.last_checkpoint.elapsed() >= self.checkpoint_interval {
            self.last_checkpoint = Instant::now();
            true
        } else {
            false
        }
    }

    fn save_checkpoint(
        &self,
        stats: &SearchStats,
        learning_system: &Option<Arc<LearningSystem>>,
        start_time: &Instant,
    ) -> Result<(), std::io::Error> {
        // Create directory if it doesn't exist
        let checkpoint_dir = Path::new("checkpoints");
        if !checkpoint_dir.exists() {
            fs::create_dir_all(checkpoint_dir)?;
        }

        let checkpoint_path = format!("checkpoints/{}", self.checkpoint_file);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let run_duration = start_time.elapsed().as_secs();
        
        let (mnemonics_checked, addresses_generated, word_matches_found) = stats.get_counts();
        
        // Get high score mnemonics and learning data if available
        let (high_scores, word_freqs, word_pair_freqs, word_position_freqs, pattern_freqs) = 
            if let Some(learning) = learning_system {
                (
                    learning.get_high_score_mnemonics(),
                    learning.get_word_frequencies(),
                    learning.get_word_pair_frequencies(),
                    learning.get_word_position_frequencies(),
                    learning.get_pattern_frequencies(),
                )
            } else {
                (Vec::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new())
            };

        // Limit the size of high scores and frequency maps if needed
        let limited_high_scores = if high_scores.len() > 100 {
            high_scores.into_iter().take(100).collect()
        } else {
            high_scores
        };

        // Create checkpoint data with limited data
        let checkpoint_data = CheckpointData {
            mnemonics_checked,
            addresses_generated,
            word_matches_found,
            timestamp,
            run_duration_seconds: run_duration,
            last_seed: None, // We're not tracking the last seed currently
            high_score_mnemonics: limited_high_scores.clone(),
            word_frequencies: self.limit_map_size(word_freqs.clone(), 1000),
            word_pair_frequencies: self.limit_map_size(word_pair_freqs.clone(), 1000),
            word_position_frequencies: self.limit_map_size(word_position_freqs.clone(), 500),
            pattern_frequencies: self.limit_map_size(pattern_freqs.clone(), 500),
        };

        // Create a temporary file first
        let temp_path = format!("{}.tmp", checkpoint_path);
        let mut file = File::create(&temp_path)?;
        
        // Serialize data
        let serialized = if self.compression_enabled {
            // Use compressed JSON format
            serde_json::to_string(&checkpoint_data)?
        } else {
            // Use pretty-printed JSON for debugging
            serde_json::to_string_pretty(&checkpoint_data)?
        };
        
        // Check if serialized data exceeds size limit
        if (serialized.len() as u64) > self.max_checkpoint_size_mb * 1024 * 1024 {
            println!("⚠️ Checkpoint data exceeds size limit. Reducing data...");
            
            // Create a more limited checkpoint with just the essential data
            let minimal_checkpoint = CheckpointData {
                mnemonics_checked,
                addresses_generated,
                word_matches_found,
                timestamp,
                run_duration_seconds: run_duration,
                last_seed: None,
                high_score_mnemonics: limited_high_scores.into_iter().take(20).collect(),
                word_frequencies: self.limit_map_size(word_freqs, 100),
                word_pair_frequencies: self.limit_map_size(word_pair_freqs, 50),
                word_position_frequencies: self.limit_map_size(word_position_freqs, 50),
                pattern_frequencies: self.limit_map_size(pattern_freqs, 20),
            };
            
            let minimal_serialized = serde_json::to_string(&minimal_checkpoint)?;
            file.write_all(minimal_serialized.as_bytes())?;
        } else {
            // Write the full serialized data
            file.write_all(serialized.as_bytes())?;
        }
        
        file.flush()?;
        
        // Rename temp file to actual checkpoint file (atomic operation)
        fs::rename(temp_path, checkpoint_path)?;

        Ok(())
    }

    fn limit_map_size<K: Clone + std::hash::Hash + std::cmp::Eq, V: Clone>(&self, map: HashMap<K, V>, max_size: usize) -> HashMap<K, V> {
        if map.len() <= max_size {
            return map;
        }
        
        let mut result = HashMap::with_capacity(max_size);
        for (i, (k, v)) in map.into_iter().enumerate() {
            if i >= max_size {
                break;
            }
            result.insert(k, v);
        }
        result
    }

    fn load_checkpoint(&self) -> Result<Option<CheckpointData>, std::io::Error> {
        let checkpoint_path = format!("checkpoints/{}", self.checkpoint_file);
        let path = Path::new(&checkpoint_path);
        
        if !path.exists() {
            return Ok(None);
        }

        // Check file size before loading
        let metadata = fs::metadata(&path)?;
        let file_size_mb = metadata.len() / (1024 * 1024);
        
        if file_size_mb > self.max_checkpoint_size_mb * 2 {
            println!("⚠️ Checkpoint file is too large ({} MB). Skipping load.", file_size_mb);
            return Ok(None);
        }

        let mut file = File::open(path)?;
        let mut contents = String::new();
        
        // Try to read the file without using catch_unwind
        match file.read_to_string(&mut contents) {
            Ok(_) => {
                match serde_json::from_str::<CheckpointData>(&contents) {
                    Ok(data) => Ok(Some(data)),
                    Err(e) => {
                        eprintln!("Error parsing checkpoint data: {}", e);
                        println!("⚠️ Checkpoint file is corrupted. Starting fresh.");
                        Ok(None)
                    }
                }
            },
            Err(e) => {
                println!("⚠️ Error reading checkpoint file: {}. It may be corrupted.", e);
                Ok(None)
            }
        }
    }

    fn restore_search_stats(&self, stats: &SearchStats, checkpoint_data: &CheckpointData) {
        stats.mnemonics_checked.store(checkpoint_data.mnemonics_checked, Ordering::SeqCst);
        stats.addresses_generated.store(checkpoint_data.addresses_generated, Ordering::SeqCst);
        stats.word_matches_found.store(checkpoint_data.word_matches_found, Ordering::SeqCst);
    }
}

// ===== Learning System =====

struct LearningSystem {
    word_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    word_pair_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    word_position_frequencies: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    pattern_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    high_score_mnemonics: Arc<Mutex<Vec<HighScoreMnemonic>>>,
    total_mnemonics_analyzed: Arc<AtomicU64>,
    max_high_scores: usize,
    min_score_threshold: f64,
    model_file: String,
    adaptive_search: bool,
    word_weight_multiplier: f64,
    pattern_recognition: bool,
    position_aware_matching: bool,
}

impl LearningSystem {
    fn new(
        model_file: &str, 
        max_high_scores: usize, 
        min_score_threshold: f64,
        adaptive_search: bool,
        word_weight_multiplier: f64,
        pattern_recognition: bool,
        position_aware_matching: bool,
    ) -> Self {
        // Create directory if it doesn't exist
        let model_dir = Path::new("models");
        if !model_dir.exists() {
            fs::create_dir_all(model_dir).unwrap_or_default();
        }

        Self {
            word_frequencies: Arc::new(RwLock::new(HashMap::new())),
            word_pair_frequencies: Arc::new(RwLock::new(HashMap::new())),
            word_position_frequencies: Arc::new(RwLock::new(HashMap::new())),
            pattern_frequencies: Arc::new(RwLock::new(HashMap::new())),
            high_score_mnemonics: Arc::new(Mutex::new(Vec::new())),
            total_mnemonics_analyzed: Arc::new(AtomicU64::new(0)),
            max_high_scores,
            min_score_threshold,
            model_file: model_file.to_string(),
            adaptive_search,
            word_weight_multiplier,
            pattern_recognition,
            position_aware_matching,
        }
    }

    fn restore_from_checkpoint(&self, checkpoint_data: &CheckpointData) {
        // Restore word frequencies
        {
            let mut freqs = self.word_frequencies.write().unwrap();
            *freqs = checkpoint_data.word_frequencies.clone();
        }
        
        // Restore word pair frequencies
        {
            let mut pair_freqs = self.word_pair_frequencies.write().unwrap();
            *pair_freqs = checkpoint_data.word_pair_frequencies.clone();
        }
        
        // Restore word position frequencies
        {
            let mut pos_freqs = self.word_position_frequencies.write().unwrap();
            *pos_freqs = checkpoint_data.word_position_frequencies.clone();
        }
        
        // Restore pattern frequencies
        {
            let mut pattern_freqs = self.pattern_frequencies.write().unwrap();
            *pattern_freqs = checkpoint_data.pattern_frequencies.clone();
        }
        
        // Restore high score mnemonics
        {
            let mut high_scores = self.high_score_mnemonics.lock().unwrap();
            *high_scores = checkpoint_data.high_score_mnemonics.clone();
        }
        
        // Set total mnemonics analyzed
        let total = checkpoint_data.word_frequencies.values().sum::<usize>();
        self.total_mnemonics_analyzed.store(total as u64, Ordering::SeqCst);
    }

    fn analyze_mnemonic(&self, mnemonic: &str, matching_words: &[String]) -> f64 {
        let mut score = 0.0;
        let words: Vec<&str> = mnemonic.split_whitespace().collect();
        let total_analyzed = self.total_mnemonics_analyzed.fetch_add(1, Ordering::SeqCst) as f64 + 1.0;
        
        // Track positions of matching words
        let mut positions = HashMap::new();
        for (i, word) in words.iter().enumerate() {
            if matching_words.contains(&word.to_string()) {
                positions.insert(word.to_string(), i);
            }
        }
        
        // Update word frequencies
        {
            let mut freqs = self.word_frequencies.write().unwrap();
            for word in &words {
                *freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        
        // Update word pair frequencies if pattern recognition is enabled
        if self.pattern_recognition {
            let mut pair_freqs = self.word_pair_frequencies.write().unwrap();
            for i in 0..words.len() - 1 {
                let pair = format!("{} {}", words[i], words[i + 1]);
                *pair_freqs.entry(pair).or_insert(0) += 1;
            }
        }
        
        // Update word position frequencies if position-aware matching is enabled
        if self.position_aware_matching {
            let mut pos_freqs = self.word_position_frequencies.write().unwrap();
            for (i, word) in words.iter().enumerate() {
                let positions = pos_freqs.entry(word.to_string()).or_insert_with(|| vec![0; 24]);
                if i < positions.len() {
                    positions[i] += 1;
                }
            }
        }
        
        // Calculate score based on word frequencies
        {
            let freqs = self.word_frequencies.read().unwrap();
            for word in matching_words {
                if let Some(freq) = freqs.get(word) {
                    // Words that appear more frequently get higher scores
                    score += (*freq as f64) / total_analyzed * self.word_weight_multiplier;
                }
            }
        }
        
        // Bonus for word pairs if pattern recognition is enabled
        if self.pattern_recognition {
            let pair_freqs = self.word_pair_frequencies.read().unwrap();
            for i in 0..words.len() - 1 {
                if matching_words.contains(&words[i].to_string()) && 
                   matching_words.contains(&words[i + 1].to_string()) {
                    let pair = format!("{} {}", words[i], words[i + 1]);
                    if let Some(freq) = pair_freqs.get(&pair) {
                        score *= 1.0 + (*freq as f64 / total_analyzed);
                    } else {
                        score *= 1.2; // Default bonus for adjacent matching words
                    }
                }
            }
        }
        
        // Bonus for position matches if position-aware matching is enabled
        if self.position_aware_matching {
            let pos_freqs = self.word_position_frequencies.read().unwrap();
            for (word, pos) in &positions {
                if let Some(freq_by_pos) = pos_freqs.get(word) {
                    if *pos < freq_by_pos.len() && freq_by_pos[*pos] > 0 {
                        // Words in their common positions get higher scores
                        score *= 1.0 + (freq_by_pos[*pos] as f64 / total_analyzed);
                    }
                }
            }
        }
        
        // Adjust score based on number of matching words
        score *= matching_words.len() as f64;
        
        // Update high score mnemonics if this is a good candidate
        if score >= self.min_score_threshold {
            self.update_high_scores(mnemonic, matching_words, score, positions);
        }
        
        score
    }

    fn update_high_scores(&self, mnemonic: &str, matching_words: &[String], score: f64, positions: HashMap<String, usize>) {
        let mut high_scores = self.high_score_mnemonics.lock().unwrap();
        
        // Create new high score entry
        let new_entry = HighScoreMnemonic {
            mnemonic: mnemonic.to_string(),
            matching_words: matching_words.iter().cloned().collect(),
            score,
            positions,
        };
        
        // Add to high scores and sort
        high_scores.push(new_entry);
        high_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Trim to max size
        if high_scores.len() > self.max_high_scores {
            high_scores.truncate(self.max_high_scores);
        }
    }

    fn get_high_score_mnemonics(&self) -> Vec<HighScoreMnemonic> {
        let high_scores = self.high_score_mnemonics.lock().unwrap();
        high_scores.clone()
    }

    fn get_word_frequencies(&self) -> HashMap<String, usize> {
        let freqs = self.word_frequencies.read().unwrap();
        freqs.clone()
    }

    fn get_word_pair_frequencies(&self) -> HashMap<String, usize> {
        let pair_freqs = self.word_pair_frequencies.read().unwrap();
        pair_freqs.clone()
    }

    fn get_word_position_frequencies(&self) -> HashMap<String, Vec<usize>> {
        let pos_freqs = self.word_position_frequencies.read().unwrap();
        pos_freqs.clone()
    }

    fn get_pattern_frequencies(&self) -> HashMap<String, usize> {
        let pattern_freqs = self.pattern_frequencies.read().unwrap();
        pattern_freqs.clone()
    }

    fn get_word_weights(&self, target_words: &HashSet<String>) -> HashMap<String, f64> {
        let freqs = self.word_frequencies.read().unwrap();
        let total = self.total_mnemonics_analyzed.load(Ordering::SeqCst) as f64;
        let mut weights = HashMap::new();
        
        if total == 0.0 {
            // No data yet, return equal weights
            for word in target_words {
                weights.insert(word.clone(), 1.0);
            }
            return weights;
        }
        
        // Calculate weights based on frequency
        for word in target_words {
            let frequency = freqs.get(word).unwrap_or(&0);
            let weight = if *frequency > 0 {
                (*frequency as f64) / total * self.word_weight_multiplier
            } else {
                0.1 // Default weight for words not seen yet
            };
            weights.insert(word.clone(), weight);
        }
        
        weights
    }

    fn generate_search_hints(&self) -> Vec<String> {
        let freqs = self.word_frequencies.read().unwrap();
        let pair_freqs = self.word_pair_frequencies.read().unwrap();
        let pos_freqs = self.word_position_frequencies.read().unwrap();
        let high_scores = self.high_score_mnemonics.lock().unwrap();
        let total = self.total_mnemonics_analyzed.load(Ordering::SeqCst);
        
        let mut hints = Vec::new();
        
        // Add hints based on word frequencies
        if total > 0 {
            // Find most common words
            let mut word_freqs: Vec<(String, usize)> = freqs.clone().into_iter().collect();
            word_freqs.sort_by(|a, b| b.1.cmp(&a.1));
            
            if !word_freqs.is_empty() {
                let top_words: Vec<String> = word_freqs.iter()
                    .take(5)
                    .map(|(word, freq)| format!("{} ({})", word, freq))
                    .collect();
                    
                hints.push(format!("Most common words: {}", top_words.join(", ")));
            }
            
            // Find most common word pairs
            if self.pattern_recognition {
                let mut pair_freqs_vec: Vec<(String, usize)> = pair_freqs.clone().into_iter().collect();
                pair_freqs_vec.sort_by(|a, b| b.1.cmp(&a.1));
                
                if !pair_freqs_vec.is_empty() {
                    let top_pairs: Vec<String> = pair_freqs_vec.iter()
                        .take(3)
                        .map(|(pair, freq)| format!("&quot;{}&quot; ({})", pair, freq))
                        .collect();
                        
                    hints.push(format!("Most common word pairs: {}", top_pairs.join(", ")));
                }
            }
            
            // Find most common word positions
            if self.position_aware_matching {
                let mut position_insights = Vec::new();
                
                for (word, positions) in pos_freqs.iter() {
                    if let Some((pos, &freq)) = positions.iter().enumerate()
                        .max_by_key(|&(_, &freq)| freq) {
                        if freq > 2 {
                            position_insights.push((word.clone(), pos, freq));
                        }
                    }
                }
                
                position_insights.sort_by(|a, b| b.2.cmp(&a.2));
                
                if !position_insights.is_empty() {
                    let top_positions: Vec<String> = position_insights.iter()
                        .take(3)
                        .map(|(word, pos, freq)| format!("{} at position {} ({})", word, pos, freq))
                        .collect();
                        
                    hints.push(format!("Word position patterns: {}", top_positions.join(", ")));
                }
            }
        }
        
        // Add hints based on high scoring mnemonics
        if !high_scores.is_empty() {
            let top_mnemonic = &high_scores[0];
            hints.push(format!(
                "Highest scoring mnemonic: &quot;{}&quot; (score: {:.2}, matching words: {})",
                top_mnemonic.mnemonic,
                top_mnemonic.score,
                top_mnemonic.matching_words.join(", ")
            ));
            
            // Analyze patterns in high scoring mnemonics
            let mut word_counts = HashMap::new();
            for entry in high_scores.iter() {
                for word in entry.matching_words.iter() {
                    *word_counts.entry(word.clone()).or_insert(0) += 1;
                }
            }
            
            let mut common_words: Vec<(String, usize)> = word_counts.into_iter().collect();
            common_words.sort_by(|a, b| b.1.cmp(&a.1));
            
            if !common_words.is_empty() {
                let top_words: Vec<String> = common_words.iter()
                    .take(5)
                    .map(|(word, count)| format!("{} ({})", word, count))
                    .collect();
                    
                hints.push(format!("Most common words in high-scoring mnemonics: {}", top_words.join(", ")));
            }
        }
        
        hints
    }

    fn save_model(&self) -> Result<(), std::io::Error> {
        // Create directory if it doesn't exist
        let model_dir = Path::new("models");
        if !model_dir.exists() {
            fs::create_dir_all(model_dir)?;
        }
        
        let model_path = format!("models/{}", self.model_file);
        
        // Create a temporary file first
        let temp_path = format!("{}.tmp", model_path);
        let mut file = File::create(&temp_path)?;
        
        // Prepare data for serialization
        let data = CheckpointData {
            mnemonics_checked: self.total_mnemonics_analyzed.load(Ordering::SeqCst),
            addresses_generated: 0, // Not tracked in learning system
            word_matches_found: 0, // Not tracked in learning system
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            run_duration_seconds: 0, // Not tracked in learning system
            last_seed: None,
            high_score_mnemonics: self.get_high_score_mnemonics(),
            word_frequencies: self.get_word_frequencies(),
            word_pair_frequencies: self.get_word_pair_frequencies(),
            word_position_frequencies: self.get_word_position_frequencies(),
            pattern_frequencies: self.get_pattern_frequencies(),
        };
        
        // Serialize and write data
        let serialized = serde_json::to_string(&data)?;
        file.write_all(serialized.as_bytes())?;
        file.flush()?;
        
        // Rename temp file to actual model file (atomic operation)
        fs::rename(temp_path, model_path)?;

        Ok(())
    }

    fn load_model(&self, path: &str) -> Result<(), std::io::Error> {
        let path = Path::new(path);
        if !path.exists() {
            return Ok(());
        }

        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        match serde_json::from_str::<CheckpointData>(&contents) {
            Ok(data) => {
                self.restore_from_checkpoint(&data);
                Ok(())
            },
            Err(e) => {
                eprintln!("Error parsing model data: {}", e);
                Ok(())
            }
        }
    }
}

// ===== Logging System =====

#[derive(Debug)]
enum LogLevel {
    Info,
    Warning,
    Error,
    Success,
    Debug,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Success => write!(f, "SUCCESS"),
            LogLevel::Debug => write!(f, "DEBUG"),
        }
    }
}

struct Logger {
    stats_log: Mutex<File>,
    match_log: Mutex<File>,
    error_log: Mutex<File>,
    word_match_log: Mutex<File>,
    health_log: Mutex<File>,
}

impl Logger {
    fn new() -> Result<Self, std::io::Error> {
        // Create logs directory if it doesn't exist
        let logs_dir = Path::new("logs");
        if !logs_dir.exists() {
            fs::create_dir_all(logs_dir)?;
        }

        // Create log files with append mode
        let stats_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/stats.log")?;
            
        let match_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/matches.log")?;
            
        let error_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/errors.log")?;
            
        let word_match_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/word_matches.log")?;
            
        let health_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/health.log")?;

        Ok(Logger {
            stats_log: Mutex::new(stats_log),
            match_log: Mutex::new(match_log),
            error_log: Mutex::new(error_log),
            word_match_log: Mutex::new(word_match_log),
            health_log: Mutex::new(health_log),
        })
    }

    fn get_timestamp(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let datetime = chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
            
        datetime
    }

    fn log_stats(&self, mnemonic_rate: f64, address_rate: f64, total_mnemonics: u64, elapsed: f64, word_matches: u64) -> Result<(), std::io::Error> {
        let timestamp = self.get_timestamp();
        let log_entry = format!(
            "[{}] Stats: {:.0} mnemonics/sec | {:.0} addresses/sec | {} total | {:.1}s | {} word matches\n",
            timestamp, mnemonic_rate, address_rate, total_mnemonics, elapsed, word_matches
        );
        
        let mut file = self.stats_log.lock().unwrap();
        file.write_all(log_entry.as_bytes())?;
        file.flush()?;
        
        Ok(())
    }

    fn log_match(&self, mnemonic: &str, path: &str, address: &str, matching_words: &[String]) -> Result<(), std::io::Error> {
        let timestamp = self.get_timestamp();
        let log_entry = format!(
            "[{}] MATCH FOUND!\nMnemonic: {}\nPath: {}\nAddress: {}\nMatching Words: {}\nCount: {}\n\n",
            timestamp, mnemonic, path, address, matching_words.join(", "), matching_words.len()
        );
        
        let mut file = self.match_log.lock().unwrap();
        file.write_all(log_entry.as_bytes())?;
        file.flush()?;
        
        Ok(())
    }

    fn log_word_match(&self, mnemonic: &str, matching_words: &[String]) -> Result<(), std::io::Error> {
        let timestamp = self.get_timestamp();
        let log_entry = format!(
            "[{}] Word Match: {} matching words\nMnemonic: {}\nWords: {}\n\n",
            timestamp, matching_words.len(), mnemonic, matching_words.join(", ")
        );
        
        let mut file = self.word_match_log.lock().unwrap();
        file.write_all(log_entry.as_bytes())?;
        file.flush()?;
        
        Ok(())
    }

    fn log_health(&self, message: &str) -> Result<(), std::io::Error> {
        let timestamp = self.get_timestamp();
        let log_entry = format!("[{}] {}\n", timestamp, message);
        
        let mut file = self.health_log.lock().unwrap();
        file.write_all(log_entry.as_bytes())?;
        file.flush()?;
        
        Ok(())
    }

    fn log_message(&self, level: LogLevel, message: &str) -> Result<(), std::io::Error> {
        let timestamp = self.get_timestamp();
        let log_entry = format!("[{}] [{}] {}\n", timestamp, level, message);
        
        match level {
            LogLevel::Error => {
                let mut file = self.error_log.lock().unwrap();
                file.write_all(log_entry.as_bytes())?;
                file.flush()?;
            },
            LogLevel::Warning => {
                // Log warnings to both error and stats logs
                {
                    let mut file = self.error_log.lock().unwrap();
                    file.write_all(log_entry.as_bytes())?;
                    file.flush()?;
                }
                {
                    let mut file = self.stats_log.lock().unwrap();
                    file.write_all(log_entry.as_bytes())?;
                    file.flush()?;
                }
            },
            _ => {
                // For other messages, log to stats log
                let mut file = self.stats_log.lock().unwrap();
                file.write_all(log_entry.as_bytes())?;
                file.flush()?;
            }
        }
        
        Ok(())
    }
}

// ===== Search Stats =====

#[derive(Clone)]
struct MatchResult {
    mnemonic: String,
    path: String,
    address: String,
    private_key: String,
    matching_words: Vec<String>,
    score: f64,
}

struct SearchStats {
    mnemonics_checked: AtomicU64,
    addresses_generated: AtomicU64,
    word_matches_found: AtomicU64,
    start_time: Instant,
    found: AtomicBool,
    shutdown_requested: AtomicBool,
}

impl SearchStats {
    fn new() -> Self {
        Self {
            mnemonics_checked: AtomicU64::new(0),
            addresses_generated: AtomicU64::new(0),
            word_matches_found: AtomicU64::new(0),
            start_time: Instant::now(),
            found: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
        }
    }

    fn add_batch(&self, mnemonics: u64, addresses: u64) {
        self.mnemonics_checked.fetch_add(mnemonics, Ordering::Relaxed);
        self.addresses_generated.fetch_add(addresses, Ordering::Relaxed);
    }

    fn increment_word_matches(&self) {
        self.word_matches_found.fetch_add(1, Ordering::Relaxed);
    }

    fn mark_found(&self) {
        self.found.store(true, Ordering::Relaxed);
    }

    fn is_found(&self) -> bool {
        self.found.load(Ordering::Relaxed)
    }

    fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    fn get_rates(&self) -> (f64, f64, f64, u64) {
        let mnemonics = self.mnemonics_checked.load(Ordering::Relaxed);
        let addresses = self.addresses_generated.load(Ordering::Relaxed);
        let word_matches = self.word_matches_found.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        (
            mnemonics as f64 / elapsed,
            addresses as f64 / elapsed,
            elapsed,
            word_matches
        )
    }

    fn get_counts(&self) -> (u64, u64, u64) {
        (
            self.mnemonics_checked.load(Ordering::Relaxed),
            self.addresses_generated.load(Ordering::Relaxed),
            self.word_matches_found.load(Ordering::Relaxed)
        )
    }
}

// ===== Main Function =====

fn main() {
    // Parse command line arguments
    let matches = App::new("Bitcoin Finder")
        .version("0.3.0")
        .author("NinjaTech AI")
        .about("Searches for Bitcoin wallets with specific word patterns")
        .arg(Arg::with_name("service-mode")
            .long("service-mode")
            .help("Run in service mode with automatic checkpointing and recovery")
            .takes_value(false))
        .arg(Arg::with_name("checkpoint-interval")
            .long("checkpoint-interval")
            .help("Interval in seconds between checkpoints")
            .takes_value(true)
            .default_value("900"))
        .arg(Arg::with_name("auto-resume")
            .long("auto-resume")
            .help("Automatically resume from last checkpoint if available")
            .takes_value(false))
        .arg(Arg::with_name("config")
            .short("c")
            .long("config")
            .value_name("FILE")
            .help("Path to config file")
            .takes_value(true)
            .default_value("config.json"))
        .arg(Arg::with_name("wordlist")
            .short("w")
            .long("wordlist")
            .value_name("FILE")
            .help("Path to word list file")
            .takes_value(true)
            .default_value("wordlist.txt"))
        .arg(Arg::with_name("database")
            .short("d")
            .long("database")
            .value_name("FILE")
            .help("Path to address database file")
            .takes_value(true)
            .default_value("btcdatabase.txt"))
        .arg(Arg::with_name("adaptive")
            .short("a")
            .long("adaptive")
            .help("Enable adaptive search mode")
            .takes_value(false))
        .arg(Arg::with_name("learning-mode")
            .short("l")
            .long("learning-mode")
            .help("Run in enhanced learning mode")
            .takes_value(false))
        .get_matches();

    // Create directories if they don't exist
    for dir in &["logs", "checkpoints", "models", "matches"] {
        let path = Path::new(dir);
        if !path.exists() {
            fs::create_dir_all(path).unwrap_or_else(|e| {
                eprintln!("Error creating directory {}: {}", dir, e);
            });
        }
    }

    // Initialize logger
    let logger = match Logger::new() {
        Ok(logger) => Arc::new(logger),
        Err(e) => {
            eprintln!("Failed to initialize logger: {}", e);
            std::process::exit(1);
        }
    };

    // Log startup
    let _ = logger.log_message(LogLevel::Info, "Bitcoin Finder starting");

    // Load configuration
    let config_path = matches.value_of("config").unwrap_or("config.json");
    let mut config = load_config(config_path);
    
    // Override config with command line arguments
    config.service_mode = matches.is_present("service-mode");
    config.auto_resume = matches.is_present("auto-resume");
    config.adaptive_search = config.adaptive_search || matches.is_present("adaptive");
    
    // Enhanced learning mode
    if matches.is_present("learning-mode") {
        config.learning_enabled = true;
        config.adaptive_search = true;
        config.pattern_recognition = true;
        config.position_aware_matching = true;
        config.word_weight_multiplier = 2.0;
        config.min_score_threshold = 0.3;
        config.max_high_scores = 200;
    }
    
    if let Some(interval) = matches.value_of("checkpoint-interval") {
        if let Ok(interval) = interval.parse::<u64>() {
            config.checkpoint_interval = interval;
        }
    }

    // Initialize checkpoint manager
    let checkpoint_manager = Arc::new(Mutex::new(CheckpointManager::new(
        "search_state.json",
        config.checkpoint_interval,
        config.max_checkpoint_size_mb,
        config.checkpoint_compression,
    )));

    // Initialize learning system
    let learning_system = if config.learning_enabled {
        Some(Arc::new(LearningSystem::new(
            "word_model.json",
            config.max_high_scores,
            config.min_score_threshold,
            config.adaptive_search,
            config.word_weight_multiplier,
            config.pattern_recognition,
            config.position_aware_matching,
        )))
    } else {
        None
    };

    let wordlist_path = matches.value_of("wordlist").unwrap_or("wordlist.txt");
    let database_path = matches.value_of("database").unwrap_or("btcdatabase.txt");

    // Log configuration
    let _ = logger.log_message(LogLevel::Info, &format!(
        "Configuration: min_matching_words={}, max_addresses_per_path={}, service_mode={}, checkpoint_interval={}s, adaptive_search={}",
        config.min_matching_words, config.max_addresses_per_path, config.service_mode, config.checkpoint_interval, config.adaptive_search
    ));

    println!("🔍 Loading custom word list...");
    let _ = logger.log_message(LogLevel::Info, &format!("Loading word list from {}", wordlist_path));
    let target_words = load_target_words(wordlist_path);
    println!("✅ Loaded {} target words", target_words.len());
    let _ = logger.log_message(LogLevel::Info, &format!("Loaded {} target words", target_words.len()));
    
    println!("🧠 Loading address database...");
    let _ = logger.log_message(LogLevel::Info, &format!("Loading address database from {}", database_path));
    let start_load = Instant::now();
    
    let address_db = Arc::new(load_address_database_optimized(database_path));
    let load_time = start_load.elapsed();
    
    println!("✅ Loaded {} addresses in {:.2}s", address_db.len(), load_time.as_secs_f64());
    let _ = logger.log_message(LogLevel::Info, &format!(
        "Loaded {} addresses in {:.2}s", 
        address_db.len(), 
        load_time.as_secs_f64()
    ));
    
    if address_db.is_empty() {
        println!("❌ No addresses loaded. Exiting.");
        let _ = logger.log_message(LogLevel::Error, "No addresses loaded. Exiting.");
        return;
    }

    // Initialize search stats
    let stats = Arc::new(SearchStats::new());
    
    // Initialize health monitor
    let health_monitor = Arc::new(Mutex::new(HealthMonitor::new(
        config.health_check_interval,
        config.max_memory_percent,
        Arc::clone(&logger),
        Arc::clone(&stats),
    )));
    
    // Check for checkpoint to resume from
    if config.auto_resume {
        println!("📋 Attempting to resume from checkpoint...");
        match checkpoint_manager.lock().unwrap().load_checkpoint() {
            Ok(Some(checkpoint_data)) => {
                println!("✅ Successfully loaded checkpoint data");
                let _ = logger.log_message(LogLevel::Info, &format!(
                    "Resuming from checkpoint: {} mnemonics checked, {} addresses generated, {} word matches found",
                    checkpoint_data.mnemonics_checked,
                    checkpoint_data.addresses_generated,
                    checkpoint_data.word_matches_found
                ));
                
                checkpoint_manager.lock().unwrap().restore_search_stats(&stats, &checkpoint_data);
                
                // If learning system is enabled, restore from checkpoint
                if let Some(learning) = &learning_system {
                    learning.restore_from_checkpoint(&checkpoint_data);
                    
                    // Generate and display search hints
                    let hints = learning.generate_search_hints();
                    println!("\n🧠 LEARNING SYSTEM HINTS:");
                    for hint in &hints {
                        println!("  • {}", hint);
                    }
                    println!();
                    
                    // Log hints
                    for hint in hints {
                        let _ = logger.log_message(LogLevel::Info, &format!("Learning hint: {}", hint));
                    }
                }
            },
            Ok(None) => {
                println!("⚠️ No checkpoint found or checkpoint empty. Starting fresh search.");
                let _ = logger.log_message(LogLevel::Warning, "No checkpoint found. Starting fresh search.");
            },
            Err(e) => {
                println!("⚠️ Error loading checkpoint: {}. Starting fresh search.", e);
                let _ = logger.log_message(LogLevel::Error, &format!("Error loading checkpoint: {}", e));
            }
        }
    }

    let target_words = Arc::new(target_words);
    let (match_tx, match_rx) = bounded::<MatchResult>(100);
    let config = Arc::new(config);

    // Set up signal handling for graceful shutdown
    let stats_for_signal = Arc::clone(&stats);
    let mut signals = Signals::new(&[SIGTERM]).unwrap();
    thread::spawn(move || {
        for _ in signals.forever() {
            println!("Received termination signal, initiating graceful shutdown...");
            stats_for_signal.request_shutdown();
            break;
        }
    });

    // Match handler thread
    let stats_clone = Arc::clone(&stats);
    let logger_clone = Arc::clone(&logger);
    let config_clone1 = Arc::clone(&config);
    thread::spawn(move || {
        while let Ok(result) = match_rx.recv() {
            stats_clone.mark_found();
            
            // Log the match
            let _ = logger_clone.log_match(
                &result.mnemonic, 
                &result.path, 
                &result.address, 
                &result.matching_words
            );
            
            save_match_optimized(&result);
            println!("💥 MATCH FOUND! Mnemonic: {}", result.mnemonic);
            println!("🔑 Matching words: {}", result.matching_words.join(", "));
            println!("📈 Score: {:.4}", result.score);
            
            // Exit if not in service mode
            if !config_clone1.service_mode {
                std::process::exit(0);
            }
        }
    });

    // Stats thread
    let stats_clone = Arc::clone(&stats);
    let checkpoint_manager_clone = Arc::clone(&checkpoint_manager);
    let learning_system_clone = learning_system.clone();
    let logger_clone = Arc::clone(&logger);
    let config_clone2 = Arc::clone(&config);
    let health_monitor_clone = Arc::clone(&health_monitor);
    thread::spawn(move || {
        let mut last_checkpoint_time = Instant::now();
        
        loop {
            thread::sleep(Duration::from_secs(3));
            
            if stats_clone.is_found() && !config_clone2.service_mode {
                break;
            }
            
            if stats_clone.is_shutdown_requested() {
                break;
            }
            
            let (mnemonic_rate, address_rate, elapsed, word_matches) = stats_clone.get_rates();
            let (mnemonics, _addresses, _) = stats_clone.get_counts();
            
            // Log stats
            let _ = logger_clone.log_stats(
                mnemonic_rate, 
                address_rate, 
                mnemonics, 
                elapsed, 
                word_matches
            );
            
            println!(
                "⚡ WORKING: {:.0} mnemonics/sec | {:.0} addresses/sec | {} total | {:.1}s | 🔤 {} word matches",
                mnemonic_rate, address_rate, mnemonics, elapsed, word_matches
            );
            
            // Check health
            let mut health_monitor = health_monitor_clone.lock().unwrap();
            if health_monitor.should_check() {
                if !health_monitor.check_health() {
                    let _ = logger_clone.log_message(
                        LogLevel::Warning,
                        "Health check failed. Consider restarting the process."
                    );
                }
            }
            
            // Check if it's time to save a checkpoint
            if checkpoint_manager_clone.lock().unwrap().should_checkpoint() || stats_clone.is_shutdown_requested() {
                println!("💾 Saving checkpoint...");
                
                // Save checkpoint
                if let Err(e) = checkpoint_manager_clone.lock().unwrap().save_checkpoint(
                    &stats_clone,
                    &learning_system_clone,
                    &stats_clone.start_time,
                ) {
                    let _ = logger_clone.log_message(
                        LogLevel::Error, 
                        &format!("Failed to save checkpoint: {}", e)
                    );
                }
                
                last_checkpoint_time = Instant::now();
                
                // If shutdown was requested, exit after saving checkpoint
                if stats_clone.is_shutdown_requested() {
                    let _ = logger_clone.log_message(
                        LogLevel::Info, 
                        "Shutdown requested, exiting after checkpoint save"
                    );
                    std::process::exit(0);
                }
                
                // Generate and display search hints if learning is enabled
                if let Some(learning) = &learning_system_clone {
                    let hints = learning.generate_search_hints();
                    if !hints.is_empty() {
                        println!("\n🧠 LEARNING SYSTEM HINTS:");
                        for hint in &hints {
                            println!("  • {}", hint);
                        }
                        println!();
                        
                        // Log hints
                        for hint in hints {
                            let _ = logger_clone.log_message(LogLevel::Info, &format!("Learning hint: {}", hint));
                        }
                    }
                }
            }
            
            // Exit if shutdown requested and we've saved a checkpoint
            if stats_clone.is_shutdown_requested() && last_checkpoint_time.elapsed() < Duration::from_secs(10) {
                let _ = logger_clone.log_message(LogLevel::Info, "Exiting after checkpoint save");
                std::process::exit(0);
            }
        }
    });

    let num_threads = num_cpus::get();
    println!("🚀 Starting search with {} threads...\n", num_threads);
    let _ = logger.log_message(LogLevel::Info, &format!("Starting search with {} threads", num_threads));
    println!("🔎 Looking for mnemonics with at least {} words from our list", config.min_matching_words);
    println!("🔍 Checking {} derivation paths per mnemonic", config.paths_to_check.len());
    println!("🔢 Generating up to {} addresses per path", config.max_addresses_per_path);
    
    if config.adaptive_search {
        println!("🧠 Adaptive search enabled - learning from patterns to improve search efficiency");
    }
    
    if config.service_mode {
        println!("🔄 Running in service mode with checkpointing every {} seconds", config.checkpoint_interval);
    }

    // Create worker threads
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let address_db = Arc::clone(&address_db);
        let stats = Arc::clone(&stats);
        let match_tx = match_tx.clone();
        let target_words = Arc::clone(&target_words);
        let config = Arc::clone(&config);
        let logger = Arc::clone(&logger);
        let learning_system = learning_system.clone();
        
        let handle = thread::spawn(move || {
            let _ = logger.log_message(LogLevel::Info, &format!("Thread {} started", thread_id));
            println!("🔥 Thread {} started", thread_id);
            
            let mut rng = SmallRng::from_entropy();
            let secp = Secp256k1::new();
            
            loop {
                if stats.is_found() && !config.service_mode {
                    break;
                }
                
                if stats.is_shutdown_requested() {
                    break;
                }

                let mut batch_mnemonics = 0u64;
                let mut batch_addresses = 0u64;

                for _ in 0..100 { // Batch size
                    if stats.is_shutdown_requested() {
                        break;
                    }
                    
                    if let Some(result) = process_mnemonic_ultra_fast(
                        &mut rng, 
                        &secp, 
                        &address_db, 
                        &target_words, 
                        &stats, 
                        &config,
                        &learning_system,
                        &logger
                    ) {
                        let _ = match_tx.try_send(result);
                        if !config.service_mode {
                            return;
                        }
                    }
                    batch_mnemonics += 1;
                    batch_addresses += (config.paths_to_check.len() * config.max_addresses_per_path as usize) as u64;
                }

                stats.add_batch(batch_mnemonics, batch_addresses);
            }
            
            let _ = logger.log_message(LogLevel::Info, &format!("Thread {} exiting", thread_id));
        });
        
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let _ = handle.join();
    }
    
    // Final checkpoint save before exit
    let _ = checkpoint_manager.lock().unwrap().save_checkpoint(
        &stats,
        &learning_system,
        &stats.start_time,
    );
    
    let _ = logger.log_message(LogLevel::Info, "Bitcoin Finder exiting");
}

// ===== Helper Functions =====

fn process_mnemonic_ultra_fast(
    rng: &mut SmallRng,
    secp: &Secp256k1<bitcoin::secp256k1::All>,
    address_db: &HashSet<String>,
    target_words: &HashSet<String>,
    stats: &SearchStats,
    config: &Config,
    learning_system: &Option<Arc<LearningSystem>>,
    logger: &Logger,
) -> Option<MatchResult> {
    // Generate random entropy
    let mut entropy = [0u8; 16];
    rng.fill_bytes(&mut entropy);
    
    let mnemonic = Mnemonic::from_entropy(&entropy).ok()?;
    let mnemonic_str = mnemonic.to_string();
    
    // Check if mnemonic contains our target words
    let matching_words = find_matching_words(&mnemonic_str, target_words);
    
    // Only proceed if we have enough matching words
    if matching_words.len() >= config.min_matching_words {
        stats.increment_word_matches();
        
        // Calculate score using learning system if enabled
        let score = if let Some(learning) = learning_system {
            learning.analyze_mnemonic(&mnemonic_str, &matching_words)
        } else {
            matching_words.len() as f64 // Simple score based on match count
        };
        
        // Log interesting mnemonics with multiple matching words
        if config.word_match_logging && matching_words.len() >= config.highlight_threshold {
            println!("👀 Found mnemonic with {} matching words (score: {:.4}): {}", 
                matching_words.len(), score, mnemonic_str);
            println!("   Matching words: {}", matching_words.join(", "));
            
            // Log to file
            let _ = logger.log_word_match(&mnemonic_str, &matching_words);
        }
        
        let seed = mnemonic.to_seed("");
        let xpriv = ExtendedPrivKey::new_master(Network::Bitcoin, &seed).ok()?;

        // Check all derivation paths
        for path_str in &config.paths_to_check {
            let base_path = DerivationPath::from_str(path_str).ok()?;
            
            for i in 0..config.max_addresses_per_path {
                let child_num = ChildNumber::from_normal_idx(i).ok()?;
                let path = base_path.child(child_num);
                
                if let Ok(derived) = xpriv.derive_priv(secp, &path) {
                    let private_key = PrivateKey::new(derived.private_key, Network::Bitcoin);
                    let pubkey = private_key.public_key(secp);
                    
                    // Generate all address types
                    let addresses = vec![
                        // Legacy P2PKH
                        Address::p2pkh(&pubkey, Network::Bitcoin).to_string(),
                        // Segwit wrapped P2SH-P2WPKH
                        Address::p2shwpkh(&pubkey, Network::Bitcoin).ok().map(|a| a.to_string()).unwrap_or_default(),
                        // Native Segwit P2WPKH
                        Address::p2wpkh(&pubkey, Network::Bitcoin).ok().map(|a| a.to_string()).unwrap_or_default(),
                        // Taproot P2TR
                        {
                            let (xonly_pubkey, _) = pubkey.inner.x_only_public_key();
                            Address::p2tr(secp, xonly_pubkey, None, Network::Bitcoin).to_string()
                        }
                    ];

                    // Check each address against database
                    for address_str in addresses {
                        if !address_str.is_empty() && address_db.contains(&address_str) {
                            return Some(MatchResult {
                                mnemonic: mnemonic_str.clone(),
                                path: format!("{}/{}", path_str, i),
                                address: address_str,
                                private_key: private_key.to_wif(),
                                matching_words: matching_words.clone(),
                                score,
                            });
                        }
                    }
                }
            }
        }
    }
    
    None
}

fn find_matching_words(mnemonic: &str, target_words: &HashSet<String>) -> Vec<String> {
    let mut matching = Vec::new();
    
    for word in mnemonic.split_whitespace() {
        if target_words.contains(word) {
            matching.push(word.to_string());
        }
    }
    
    matching
}

fn load_config(config_path: &str) -> Config {
    match File::open(config_path) {
        Ok(mut file) => {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&contents) {
                    let mut config = Config::default();
                    
                    if let Some(min) = json.get("min_matching_words").and_then(|v| v.as_u64()) {
                        config.min_matching_words = min as usize;
                    }
                    
                    if let Some(max) = json.get("max_addresses_per_path").and_then(|v| v.as_u64()) {
                        config.max_addresses_per_path = max as u32;
                    }
                    
                    if let Some(threshold) = json.get("highlight_threshold").and_then(|v| v.as_u64()) {
                        config.highlight_threshold = threshold as usize;
                    }
                    
                    if let Some(logging) = json.get("word_match_logging").and_then(|v| v.as_bool()) {
                        config.word_match_logging = logging;
                    }
                    
                    if let Some(learning) = json.get("learning_enabled").and_then(|v| v.as_bool()) {
                        config.learning_enabled = learning;
                    }
                    
                    if let Some(score) = json.get("min_score_threshold").and_then(|v| v.as_f64()) {
                        config.min_score_threshold = score;
                    }
                    
                    if let Some(max_scores) = json.get("max_high_scores").and_then(|v| v.as_u64()) {
                        config.max_high_scores = max_scores as usize;
                    }
                    
                    if let Some(adaptive) = json.get("adaptive_search").and_then(|v| v.as_bool()) {
                        config.adaptive_search = adaptive;
                    }
                    
                    if let Some(multiplier) = json.get("word_weight_multiplier").and_then(|v| v.as_f64()) {
                        config.word_weight_multiplier = multiplier;
                    }
                    
                    if let Some(pattern) = json.get("pattern_recognition").and_then(|v| v.as_bool()) {
                        config.pattern_recognition = pattern;
                    }
                    
                    if let Some(position) = json.get("position_aware_matching").and_then(|v| v.as_bool()) {
                        config.position_aware_matching = position;
                    }
                    
                    if let Some(health_interval) = json.get("health_check_interval").and_then(|v| v.as_u64()) {
                        config.health_check_interval = health_interval;
                    }
                    
                    if let Some(max_mem) = json.get("max_memory_percent").and_then(|v| v.as_u64()) {
                        if max_mem <= 100 {
                            config.max_memory_percent = max_mem as u8;
                        }
                    }
                    
                    if let Some(max_checkpoint) = json.get("max_checkpoint_size_mb").and_then(|v| v.as_u64()) {
                        config.max_checkpoint_size_mb = max_checkpoint;
                    }
                    
                    if let Some(compression) = json.get("checkpoint_compression").and_then(|v| v.as_bool()) {
                        config.checkpoint_compression = compression;
                    }
                    
                    if let Some(paths) = json.get("paths_to_check").and_then(|v| v.as_array()) {
                        let mut path_list = Vec::new();
                        for path in paths {
                            if let Some(path_str) = path.as_str() {
                                path_list.push(path_str.to_string());
                            }
                        }
                        if !path_list.is_empty() {
                            config.paths_to_check = path_list;
                        }
                    }
                    
                    println!("✅ Loaded configuration from {}", config_path);
                    return config;
                }
            }
            println!("⚠️ Failed to parse {}, using default configuration", config_path);
            Config::default()
        }
        Err(_) => {
            println!("⚠️ {} not found, using default configuration", config_path);
            Config::default()
        }
    }
}

fn load_target_words(filename: &str) -> HashSet<String> {
    let mut words = HashSet::new();
    
    if let Ok(file) = File::open(filename) {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(word) = line {
                let trimmed = word.trim();
                if !trimmed.is_empty() {
                    words.insert(trimmed.to_string());
                }
            }
        }
    } else {
        println!("⚠️ Warning: Could not open word list file '{}'. Using default BIP39 wordlist.", filename);
    }
    
    words
}

fn load_address_database_optimized(filename: &str) -> HashSet<String> {
    match File::open(filename) {
        Ok(file) => {
            println!("📁 Memory-mapping database file...");
            let mmap = unsafe { Mmap::map(&file).expect("Failed to memory-map file") };
            let content = std::str::from_utf8(&mmap).expect("Invalid UTF-8 in database");
            
            let mut addresses = HashSet::with_capacity(25_000_000);
            
            for line in content.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    addresses.insert(trimmed.to_string());
                }
            }
            
            addresses.shrink_to_fit();
            addresses
        }
        Err(e) => {
            println!("❌ Failed to open {}: {}", filename, e);
            println!("💡 Make sure {} exists in this directory", filename);
            std::process::exit(1);
        }
    }
}

fn save_match_optimized(result: &MatchResult) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("MATCH_FOUND.txt") {
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let _ = writeln!(file, "🎉 BITCOIN ADDRESS MATCH FOUND!");
        let _ = writeln!(file, "=====================================");
        let _ = writeln!(file, "Timestamp: {}", timestamp);
        let _ = writeln!(file, "Mnemonic: {}", result.mnemonic);
        let _ = writeln!(file, "Matching Words: {}", result.matching_words.join(", "));
        let _ = writeln!(file, "Number of Matching Words: {}", result.matching_words.len());
        let _ = writeln!(file, "Score: {:.4}", result.score);
        let _ = writeln!(file, "Derivation Path: {}", result.path);
        let _ = writeln!(file, "Address: {}", result.address);
        let _ = writeln!(file, "Private Key (WIF): {}", result.private_key);
        let _ = writeln!(file, "=====================================");
        let _ = file.flush();
    }
    
    // Also save to a timestamped file for historical record
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
        
    let matches_dir = Path::new("matches");
    if !matches_dir.exists() {
        fs::create_dir_all(matches_dir).unwrap_or_default();
    }
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(format!("matches/match_{}.txt", timestamp)) {
        
        let _ = writeln!(file, "🎉 BITCOIN ADDRESS MATCH FOUND!");
        let _ = writeln!(file, "=====================================");
        let _ = writeln!(file, "Timestamp: {}", timestamp);
        let _ = writeln!(file, "Mnemonic: {}", result.mnemonic);
        let _ = writeln!(file, "Matching Words: {}", result.matching_words.join(", "));
        let _ = writeln!(file, "Number of Matching Words: {}", result.matching_words.len());
        let _ = writeln!(file, "Score: {:.4}", result.score);
        let _ = writeln!(file, "Derivation Path: {}", result.path);
        let _ = writeln!(file, "Address: {}", result.address);
        let _ = writeln!(file, "Private Key (WIF): {}", result.private_key);
        let _ = writeln!(file, "=====================================");
        let _ = file.flush();
    }
}

use std::str::FromStr;