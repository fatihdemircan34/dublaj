class PyannoteASRPipeline:
    """Advanced ASR + Diarization Pipeline using Pyannote-audio"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfig()
        self._setup_device()
        self._initialize_models()

    def _setup_device(self):
        """Setup computation device"""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Using CPU")

    def _initialize_models(self):
        """Initialize ASR and diarization models"""
        print("📦 Loading models...")

        # Load WhisperX model
        compute_type = "float16" if self.device.type == "cuda" else "int8"
        self.whisper_model = whisperx.load_model(
            self.config.whisper_model,
            device=str(self.device),
            compute_type=compute_type,
            threads=self.config.num_threads
        )
        print(f"✅ WhisperX {self.config.whisper_model} loaded")

        # Load Pyannote diarization pipeline
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable required for pyannote models")

        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config.diarization_model,
            use_auth_token=hf_token
        ).to(self.device)

        # (İsteğe bağlı) pipeline ayarı
        try:
            self.diarization_pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": 0.58
                }
            })
        except Exception:
            # bazı versiyonlarda instantiate parametreleri değişik olabilir
            pass

        print(f"✅ Pyannote {self.config.diarization_model} loaded")

        self.align_model = None
        self.align_metadata = None

    # -------------------------
    # ASR Processing
    # -------------------------
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe audio with word-level timestamps"""
        print("🎤 Transcribing audio...")

        # Transcribe with WhisperX
        result = self.whisper_model.transcribe(
            audio_path,
            batch_size=self.config.batch_size,
            language=language,
        )

        detected_language = result["language"]
        print(f"   Language: {detected_language}")

        # Load alignment model if needed
        if self.align_model is None or self.align_metadata is None:
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=str(self.device)
            )

        # Perform word alignment
        print("🎯 Aligning words...")
        aligned = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio_path,
            str(self.device),
            return_char_alignments=False
        )

        # Extract words
        words = self._extract_words(aligned)
        print(f"✅ Transcribed {len(words)} words")

        return {
            "words": words,
            "segments": aligned["segments"],
            "language": detected_language
        }

    def _extract_words(self, aligned_result: Dict) -> List[Word]:
        """Extract Word objects from aligned transcription"""
        words = []
        for segment in aligned_result["segments"]:
            for word_data in segment.get("words", []):
                if word_data.get("start") is not None and word_data.get("end") is not None:
                    word = Word(
                        text=word_data["word"].strip(),
                        start=float(word_data["start"]),
                        end=float(word_data["end"]),
                        confidence=float(word_data.get("probability", 0.99))
                    )
                    words.append(word)
        return words

    # -------------------------
    # Diarization Processing
    # -------------------------
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> Annotation:
        """Perform speaker diarization"""
        print("👥 Performing speaker diarization...")

        # Prepare parameters
        params = {}
        if num_speakers:
            params["num_speakers"] = num_speakers
        elif self.config.min_speakers and self.config.max_speakers:
            params["min_speakers"] = self.config.min_speakers
            params["max_speakers"] = self.config.max_speakers

        # Run diarization with progress hook
        with ProgressHook() as hook:
            diarization = self.diarization_pipeline(
                audio_path,
                hook=hook,
                **params
            )

        # Get speaker statistics
        speakers = diarization.labels()
        print(f"✅ Found {len(speakers)} speakers: {sorted(speakers)}")
        return diarization

    # -------------------------
    # Speaker Assignment Methods
    # -------------------------
    def assign_speakers(self, words: List[Word], diarization: Annotation) -> List[Word]:
        """Multi-method speaker assignment with ensemble voting"""
        print("🔗 Assigning speakers to words...")

        # Convert diarization to list for easier processing
        diar_segments = [(segment, speaker) for segment, _, speaker in diarization.itertracks(yield_label=True)]

        # Apply multiple assignment methods
        for i, word in enumerate(words):
            # Method 1: Direct overlap
            overlap_scores = self._calculate_overlaps(word, diar_segments)

            # Method 2: Weighted center
            center_speaker = self._assign_by_center(word, diar_segments)

            # Method 3: Temporal context
            context_speaker = self._assign_by_context(word, words, i, diar_segments)

            # Ensemble voting (sert ağırlıklarla)
            speaker, confidence, method = self._ensemble_vote(
                overlap_scores, center_speaker, context_speaker
            )

            word.speaker = speaker
            word.speaker_confidence = confidence
            word.assignment_method = method

            # Store alternatives
            if overlap_scores:
                word.alternatives = [(s, c) for s, c, _ in overlap_scores[1:] if s != speaker][:3]

            # Check for overlaps
            word.is_overlap = self._check_overlap(overlap_scores)

        # Post-processing refinements
        words = self._refine_assignments(words)
        words = self._smooth_transitions(words)

        assigned_count = sum(1 for w in words if w.speaker)
        print(f"✅ Assigned speakers to {assigned_count}/{len(words)} words")
        return words

    def _calculate_overlaps(self, word: Word, segments: List[Tuple[Segment, str]]) -> List[Tuple[str, float, str]]:
        """Calculate overlap scores for all speakers"""
        scores = []
        for segment, speaker in segments:
            overlap_start = max(word.start, segment.start)
            overlap_end = min(word.end, segment.end)
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlap_ratio = overlap_duration / max(1e-9, word.duration)
                if overlap_ratio >= 0.9:
                    overlap_type = "full"
                elif overlap_ratio >= 0.5:
                    overlap_type = "major"
                else:
                    overlap_type = "minor"
                scores.append((speaker, overlap_ratio, overlap_type))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _assign_by_center(self, word: Word, segments: List[Tuple[Segment, str]]) -> Tuple[Optional[str], float]:
        """Assign speaker based on word center point"""
        center = word.center
        for segment, speaker in segments:
            if segment.start <= center <= segment.end:
                edge_distance = min(center - segment.start, segment.end - center)
                confidence = min(1.0, edge_distance / max(1e-9, (segment.duration / 2)))
                return speaker, confidence
        return None, 0.0

    def _assign_by_context(self, word: Word, all_words: List[Word], index: int,
                           segments: List[Tuple[Segment, str]]) -> Tuple[Optional[str], float]:
        """Assign speaker based on temporal context"""
        window = self.config.context_window
        start_idx = max(0, index - window)
        end_idx = min(len(all_words), index + window + 1)
        context_words = all_words[start_idx:end_idx]

        overlap_scores = self._calculate_overlaps(word, segments)
        if not overlap_scores:
            return None, 0.0

        primary_speaker = overlap_scores[0][0]
        primary_score = overlap_scores[0][1]

        context_speakers = [w.speaker for w in context_words if w.speaker and w != word]
        if context_speakers:
            speaker_freq = context_speakers.count(primary_speaker) / len(context_speakers)
            context_bonus = speaker_freq * 0.2
            return primary_speaker, min(1.0, primary_score + context_bonus)

        return primary_speaker, primary_score

    def _ensemble_vote(self, overlap_scores: List, center_result: Tuple,
                       context_result: Tuple) -> Tuple[str, float, str]:
        """Ensemble voting for final speaker assignment (sert ağırlıklar)"""
        votes: Dict[str, float] = {}
        weights = {"overlap": 0.7, "center": 0.3, "context": 0.0}

        # Overlap
        if overlap_scores:
            s, sc, _ = overlap_scores[0]
            votes[s] = votes.get(s, 0.0) + sc * weights["overlap"]

        # Center
        if center_result[0]:
            s, sc = center_result
            votes[s] = votes.get(s, 0.0) + sc * weights["center"]

        # Context
        if context_result[0] and weights["context"] > 0:
            s, sc = context_result
            votes[s] = votes.get(s, 0.0) + sc * weights["context"]

        if not votes:
            return "SPEAKER_00", 0.0, "fallback"

        best = max(votes.keys(), key=lambda k: votes[k])
        conf = float(votes[best])

        # method tag (bilgi amaçlı)
        method = "overlap"
        if center_result[0] == best and (not overlap_scores or votes[best] == center_result[1] * weights["center"]):
            method = "center"
        return best, conf, method

    def _check_overlap(self, overlap_scores: List) -> bool:
        """Check if word is in overlap region"""
        if len(overlap_scores) < 2:
            return False
        primary = overlap_scores[0][1]
        secondary = overlap_scores[1][1]
        return secondary >= 0.2 and abs(primary - secondary) < 0.5

    def _refine_assignments(self, words: List[Word]) -> List[Word]:
        """Refine low-confidence assignments"""
        for w in words:
            if w.speaker_confidence < self.config.confidence_threshold and w.alternatives:
                alt_speaker, alt_score = w.alternatives[0]
                if alt_score > w.speaker_confidence * 1.5:
                    w.speaker = alt_speaker
                    w.speaker_confidence = alt_score
                    w.assignment_method = "refined"
        return words

    def _smooth_transitions(self, words: List[Word]) -> List[Word]:
        """Smooth speaker transitions using temporal consistency"""
        window = 3
        for i in range(len(words)):
            if words[i].speaker_confidence < 0.6:
                start_idx = max(0, i - window)
                end_idx = min(len(words), i + window + 1)
                context = words[start_idx:end_idx]
                speaker_counts: Dict[str, int] = {}
                for w in context:
                    if w.speaker and w is not words[i]:
                        speaker_counts[w.speaker] = speaker_counts.get(w.speaker, 0) + 1
                if speaker_counts:
                    common_speaker = max(speaker_counts.keys(), key=speaker_counts.get)
                    if speaker_counts[common_speaker] >= len(context) * 0.6:
                        words[i].speaker = common_speaker
                        words[i].speaker_confidence = 0.7
                        words[i].assignment_method = "smoothed"
        return words

    # -------------------------
    # STRICT: force assignment by diarization center
    # -------------------------
    def _label_at_center(self, diarization: Annotation, t: float) -> Optional[str]:
        """Return the active label at instant t using a 1ms probe."""
        probe = Segment(t, t + 0.001)
        cropped = diarization.crop(probe)
        if cropped is None or len(cropped.labels()) == 0:
            return None
        best_label, best_dur = None, 0.0
        for seg, _, label in cropped.itertracks(yield_label=True):
            dur = seg.duration
            if dur > best_dur:
                best_dur = dur
                best_label = label
        return best_label

    def force_strict_by_diarization(self, words: List[Word], diarization: Annotation,
                                    boundary_tol: float = 0.08) -> List[Word]:
        """
        Hard-assign: set each word's speaker to the diarization label active at its center time.
        """
        strict = []
        for w in words:
            center = w.center
            spk = self._label_at_center(diarization, center)
            if spk is None:
                # find nearest diar segment label
                nearest, best_d = None, float("inf")
                for seg, _, label in diarization.itertracks(yield_label=True):
                    if center < seg.start:
                        d = seg.start - center
                    elif center > seg.end:
                        d = center - seg.end
                    else:
                        d = 0.0
                    if d < best_d:
                        best_d = d
                        nearest = label
                spk = nearest or (w.speaker or "SPEAKER_00")

            w.speaker = spk
            w.speaker_confidence = max(w.speaker_confidence, 0.75)
            w.assignment_method = "strict_diar_center"
            strict.append(w)
        return strict

    # -------------------------
    # Timeline Generation
    # -------------------------
    def create_timeline(self, words: List[Word], diarization: Optional[Annotation] = None) -> List[SpeakerTurn]:
        """
        Create speaker timeline; cut turns also at diarization boundaries.
        """
        if not words:
            return []

        diar_boundaries: List[float] = []
        if diarization is not None:
            for seg, _, _ in diarization.itertracks(yield_label=True):
                diar_boundaries.append(seg.start)
                diar_boundaries.append(seg.end)
            diar_boundaries = sorted(set(diar_boundaries))

        def must_cut_at_boundary(prev_end: float, next_start: float) -> bool:
            if not diar_boundaries:
                return False
            for b in diar_boundaries:
                if prev_end <= b <= next_start:
                    return True
            return False

        timeline: List[SpeakerTurn] = []
        current: Optional[SpeakerTurn] = None
        gap = self.config.merge_gap

        for w in words:
            new_turn = (
                    current is None
                    or w.speaker != current.speaker
                    or w.start > current.end + gap
                    or (current is not None and must_cut_at_boundary(current.end, w.start))
            )
            if new_turn:
                if current:
                    timeline.append(current)
                current = SpeakerTurn(
                    speaker=w.speaker or "UNKNOWN",
                    start=w.start,
                    end=w.end,
                    words=[w]
                )
            else:
                current.end = w.end
                current.words.append(w)

        if current:
            timeline.append(current)

        # turn confidence
        for t in timeline:
            if t.words:
                confs = [x.speaker_confidence for x in t.words if x.speaker_confidence > 0]
                t.confidence = float(np.mean(confs)) if confs else 0.0

        return timeline

    # -------------------------
    # Overlap Detection
    # -------------------------
    def detect_overlaps(self, words: List[Word]) -> List[Dict]:
        """Detect overlap regions"""
        overlaps: List[Dict] = []
        overlap_words = [w for w in words if w.is_overlap]
        if not overlap_words:
            return overlaps

        current_overlap = None
        for word in sorted(overlap_words, key=lambda x: x.start):
            if current_overlap is None:
                current_overlap = {
                    "start": word.start,
                    "end": word.end,
                    "words": [word],
                    "speakers": {word.speaker}
                }
                for alt_speaker, _ in word.alternatives:
                    current_overlap["speakers"].add(alt_speaker)
            elif word.start <= current_overlap["end"] + 0.1:
                current_overlap["end"] = max(current_overlap["end"], word.end)
                current_overlap["words"].append(word)
                current_overlap["speakers"].add(word.speaker)
                for alt_speaker, _ in word.alternatives:
                    current_overlap["speakers"].add(alt_speaker)
            else:
                if len(current_overlap["speakers"]) >= 2:
                    overlaps.append(self._format_overlap(current_overlap))
                current_overlap = {
                    "start": word.start,
                    "end": word.end,
                    "words": [word],
                    "speakers": {word.speaker}
                }
                for alt_speaker, _ in word.alternatives:
                    current_overlap["speakers"].add(alt_speaker)

        if current_overlap and len(current_overlap["speakers"]) >= 2:
            overlaps.append(self._format_overlap(current_overlap))
        return overlaps

    def _format_overlap(self, overlap_data: Dict) -> Dict:
        """Format overlap data for output"""
        return {
            "start": round(overlap_data["start"], 3),
            "end": round(overlap_data["end"], 3),
            "duration": round(overlap_data["end"] - overlap_data["start"], 3),
            "speakers": sorted(list(overlap_data["speakers"])),
            "text": " ".join(w.text for w in overlap_data["words"]),
            "word_count": len(overlap_data["words"])
        }

    # -------------------------
    # Statistics
    # -------------------------
    def calculate_statistics(self, result: DiarizationResult) -> Dict:
        """Calculate comprehensive statistics"""
        words = result.words
        turns = result.turns

        total_duration = max([w.end for w in words]) if words else 0.0
        speakers = sorted(set(w.speaker for w in words if w.speaker))

        high_conf = sum(1 for w in words if w.speaker_confidence > 0.8)
        med_conf = sum(1 for w in words if 0.5 <= w.speaker_confidence <= 0.8)
        low_conf = sum(1 for w in words if w.speaker_confidence < 0.5)

        method_counts: Dict[str, int] = {}
        for w in words:
            method_counts[w.assignment_method] = method_counts.get(w.assignment_method, 0) + 1

        speaker_stats: Dict[str, Dict] = {}
        for spk in speakers:
            spk_words = [w for w in words if w.speaker == spk]
            spk_turns = [t for t in turns if t.speaker == spk]
            speaker_stats[spk] = {
                "word_count": len(spk_words),
                "turn_count": len(spk_turns),
                "total_duration": sum(t.duration for t in spk_turns),
                "avg_confidence": float(np.mean([w.speaker_confidence for w in spk_words])) if spk_words else 0.0
            }

        return {
            "total_duration": round(total_duration, 2),
            "total_words": len(words),
            "total_turns": len(turns),
            "total_speakers": len(speakers),
            "speakers": speakers,
            "overlap_count": len(result.overlaps),
            "overlap_words": sum(1 for w in words if w.is_overlap),
            "confidence_distribution": {
                "high": high_conf,
                "medium": med_conf,
                "low": low_conf,
                "high_percentage": round(100 * high_conf / len(words), 1) if words else 0
            },
            "assignment_methods": method_counts,
            "speaker_statistics": speaker_stats,
            "avg_turn_duration": round(np.mean([t.duration for t in turns]), 2) if turns else 0,
            "avg_words_per_turn": round(np.mean([len(t.words) for t in turns]), 1) if turns else 0
        }

    # -------------------------
    # Saving
    # -------------------------
    def _sanitize_ortho(self, text: str, limit: int = 120) -> str:
        """RTTM ORT alanı için güvenli kısaltma/temizleme."""
        if text is None or not text.strip():
            return "<NA>"
        t = " ".join(text.strip().split())
        t = t.replace(" ", "_")
        if len(t) > limit:
            t = t[:limit] + "…"
        return t

    def _write_full_rttm(self, audio_path: str, turns: List[SpeakerTurn], out_dir: Path,
                         fill_ortho: bool = True, stype: str = "speech", channel: int = 1):
        """
        SPEAKER satırlarını *tüm alanlar dolu* şekilde yazar.
        SPEAKER <FILE> <CHAN> <BT> <DUR> <ORT> <STYPE> <NAME> <CONF> <SLAT>
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        file_id = Path(audio_path).stem
        rttm_path = out_dir / f"{file_id}.rttm"

        with open(rttm_path, "w", encoding="utf-8") as f:
            for turn in turns:
                bt = float(turn.start)
                dur = max(0.0, float(turn.end - turn.start))
                name = str(turn.speaker or "SPEAKER_00")
                conf = float(max(0.0, min(1.0, turn.confidence if np.isfinite(turn.confidence) else 0.0)))
                ort = self._sanitize_ortho(turn.text) if fill_ortho else "<NA>"
                st = stype or "speech"
                slat = "<NA>"
                f.write(f"SPEAKER {file_id} {channel} {bt:.3f} {dur:.3f} {ort} {st} {name} {conf:.3f} {slat}\n")

        print(f"📝 FULL RTTM kaydedildi: {rttm_path}")

    def _save_results(self, result: DiarizationResult, output_path: Path, audio_path: str):
        """Save all results to disk"""
        print("\n💾 Saving results...")

        basename = Path(audio_path).stem

        # Timeline (JSONL)
        if self.config.save_timeline:
            timeline_file = output_path / f"{basename}_timeline.jsonl"
            with open(timeline_file, "w", encoding="utf-8") as f:
                for turn in result.turns:
                    f.write(json.dumps({
                        "speaker": turn.speaker,
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                        "duration": round(turn.duration, 3),
                        "text": turn.text,
                        "confidence": round(turn.confidence, 3),
                        "word_count": len(turn.words)
                    }, ensure_ascii=False) + "\n")
            print(f"   ✓ Timeline: {timeline_file}")

        # Words (JSONL)
        words_file = output_path / f"{basename}_words.jsonl"
        with open(words_file, "w", encoding="utf-8") as f:
            for word in result.words:
                f.write(json.dumps({
                    "text": word.text,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "speaker": word.speaker,
                    "confidence": round(word.confidence, 3),
                    "speaker_confidence": round(word.speaker_confidence, 3),
                    "is_overlap": word.is_overlap,
                    "method": word.assignment_method
                }, ensure_ascii=False) + "\n")
        print(f"   ✓ Words: {words_file}")

        # Overlaps (JSON)
        if result.overlaps:
            overlaps_file = output_path / f"{basename}_overlaps.json"
            with open(overlaps_file, "w", encoding="utf-8") as f:
                json.dump(result.overlaps, f, indent=2, ensure_ascii=False)
            print(f"   ✓ Overlaps: {overlaps_file}")

        # Statistics (JSON)
        stats_file = output_path / f"{basename}_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(result.statistics, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Statistics: {stats_file}")

        # RTTM format (full)
        if self.config.save_rttm and result.turns:
            self._write_full_rttm(
                audio_path=audio_path,
                turns=result.turns,
                out_dir=output_path,
                fill_ortho=True,      # turn metnini ORT alanına yaz
                stype="speech",
                channel=1
            )

        # Segments (txt)
        if self.config.save_segments:
            segments_file = output_path / f"{basename}_segments.txt"
            with open(segments_file, "w", encoding="utf-8") as f:
                for turn in result.turns:
                    f.write(f"[{turn.start:.1f}s - {turn.end:.1f}s] {turn.speaker}: {turn.text}\n")
            print(f"   ✓ Segments: {segments_file}")

    def _print_summary(self, result: DiarizationResult):
        """Print processing summary"""
        stats = result.statistics
        print(f"\n{'='*60}")
        print("📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Duration:        {stats['total_duration']:.1f} seconds")
        print(f"Words:           {stats['total_words']}")
        print(f"Turns:           {stats['total_turns']}")
        print(f"Speakers:        {stats['total_speakers']} {stats['speakers']}")
        print(f"Overlaps:        {stats['overlap_count']} regions, {stats['overlap_words']} words")
        print(f"High Confidence: {stats['confidence_distribution']['high_percentage']:.1f}%")
        print(f"\nSpeaker Distribution:")
        for speaker, sstats in stats['speaker_statistics'].items():
            print(f"  {speaker}: {sstats['word_count']} words, "
                  f"{sstats['total_duration']:.1f}s, "
                  f"confidence: {sstats['avg_confidence']:.2f}")
        print(f"{'='*60}\n")

    # -------------------------
    # Main Processing
    # -------------------------
    def process(self, audio_path: str, output_dir: str = "output",
                language: Optional[str] = None, num_speakers: Optional[int] = None) -> DiarizationResult:
        """Main processing pipeline"""
        print(f"\n{'='*60}")
        print(f"🚀 Processing: {Path(audio_path).name}")
        print(f"{'='*60}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1) Transcribe
        transcription = self.transcribe(audio_path, language)
        words = transcription["words"]

        # 2) Diarize
        diarization = self.diarize(audio_path, num_speakers)

        # 3) Assign (soft)
        words = self.assign_speakers(words, diarization)

        # 4) STRICT hard-assign by diarization center
        words = self.force_strict_by_diarization(words, diarization, boundary_tol=0.08)

        # 5) Timeline (cut on diar boundaries)
        timeline = self.create_timeline(words, diarization=diarization)

        # 6) Overlaps
        overlaps = self.detect_overlaps(words)

        # Build result
        result = DiarizationResult(
            turns=timeline,
            words=words,
            overlaps=overlaps,
            statistics={},
            raw_diarization=diarization,
            raw_transcription=transcription
        )

        # 7) Stats
        result.statistics = self.calculate_statistics(result)

        # 8) Save
        self._save_results(result, output_path, audio_path)

        # 9) Summary
        self._print_summary(result)

        return result