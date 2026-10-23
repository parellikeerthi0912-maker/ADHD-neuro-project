import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Brain,
  Sparkles,
  Layers,
  Cpu,
  ChevronDown,
  FastForward,
  Activity,
  Play,
  ShieldCheck,
  Info,
  X,
  Target,
  Atom,
} from 'lucide-react';
import { BrainCanvas } from '../components/brain/BrainCanvas';
import { BrainExplorer } from '../components/brain/BrainExplorer';
import { UnderstandAdhdPanel } from '../components/landing/UnderstandAdhdPanel';
import { FeatureCardsSection } from '../components/landing/FeatureCardsSection';
import { EegChannelsSection } from '../components/landing/EegChannelsSection';
import { PredictionWorkflowSection } from '../components/landing/PredictionWorkflowSection';
import { ModelsSection } from '../components/landing/ModelsSection';
import { Navbar } from '../components/layout/Navbar';
import { Footer } from '../components/layout/Footer';
import { Button } from '../components/ui/Button';
import { BrainRegionKey } from '../types';
import { BRAIN_REGIONS } from '../utils/sampleData';

export const LandingPage: React.FC = () => {
  const [introStep, setIntroStep] = useState<number>(0);
  const [hasSkippedIntro, setHasSkippedIntro] = useState(false);
  const [activeHeroLobe, setActiveHeroLobe] = useState<BrainRegionKey | null>(null);

  // Cinematic opening sequence
  useEffect(() => {
    if (hasSkippedIntro) {
      setIntroStep(7);
      return;
    }

    const timers = [
      setTimeout(() => setIntroStep(1), 300),   // particles
      setTimeout(() => setIntroStep(2), 800),   // neural connections
      setTimeout(() => setIntroStep(3), 1400),  // glowing brain emerges
      setTimeout(() => setIntroStep(4), 2200),  // light pulse moves
      setTimeout(() => setIntroStep(5), 3000),  // title fades in
      setTimeout(() => setIntroStep(6), 3800),  // subtitle & labels
      setTimeout(() => setIntroStep(7), 4500),  // full interactive layout
    ];

    return () => timers.forEach(clearTimeout);
  }, [hasSkippedIntro]);

  const handleSkipIntro = () => {
    setHasSkippedIntro(true);
    setIntroStep(7);
  };

  return (
    <div className="relative min-h-screen bg-transparent text-slate-100 flex flex-col overflow-x-hidden">
      <Navbar />

      {/* Skip Intro Floating Button */}
      {!hasSkippedIntro && introStep < 7 && (
        <button
          onClick={handleSkipIntro}
          className="fixed top-24 right-6 z-50 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-700 text-xs font-mono text-cyan-300 hover:text-white hover:border-cyan-400 transition-all backdrop-blur-md"
        >
          <FastForward className="w-3.5 h-3.5" />
          <span>Skip Intro</span>
        </button>
      )}

      {/* ========================================================= */}
      {/* 1. CINEMATIC 3-COLUMN HERO SECTION (Matches Reference Image) */}
      {/* ========================================================= */}
      <section className="relative min-h-screen flex flex-col justify-center pt-24 pb-8 px-4 md:px-6 xl:px-8 overflow-hidden">
        {/* Ambient Atmosphere & Radial Glows */}
        <div className="absolute inset-0 pointer-events-none neural-grid opacity-25" />
        <div className="absolute top-1/3 left-1/4 w-[500px] h-[500px] bg-cyan-500/10 rounded-full filter blur-[140px] pointer-events-none" />
        <div className="absolute top-1/2 right-1/4 w-[500px] h-[500px] bg-violet-600/10 rounded-full filter blur-[140px] pointer-events-none" />

        {/* 3-Column Grid */}
        <div className="max-w-[1600px] mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-6 items-center relative z-10">
          
          {/* ======================================================= */}
          {/* COLUMN 1 (LEFT, ~30%): HERO TYPOGRAPHY & ACTIONS */}
          {/* ======================================================= */}
          <div className="lg:col-span-4 xl:col-span-3 space-y-6 text-center lg:text-left order-1">
            {/* Top Badge */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: introStep >= 1 ? 1 : 0, y: introStep >= 1 ? 0 : -10 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono tracking-widest uppercase shadow-[0_0_15px_rgba(0,229,255,0.2)]"
            >
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>NEUROSCIENCE × ARTIFICIAL INTELLIGENCE</span>
            </motion.div>

            {/* Main Title & Subtitle */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: introStep >= 5 ? 1 : 0, y: introStep >= 5 ? 0 : 20 }}
              transition={{ duration: 0.8 }}
              className="space-y-3"
            >
              <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold font-heading text-white tracking-tight leading-[1.05]">
                NEUROSCAN <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-400">AI</span>
              </h1>
              <p className="text-base sm:text-lg font-heading font-medium text-slate-300 leading-snug">
                EEG-Based ADHD Detection Using a Multi-Model Deep Learning Framework
              </p>
            </motion.div>

            {/* Description */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: introStep >= 6 ? 1 : 0 }}
              transition={{ duration: 0.8 }}
              className="text-xs sm:text-sm text-slate-400 leading-relaxed font-sans"
            >
              Explore EEG signal patterns through an AI-powered research prototype combining deep learning and spectral analysis.
            </motion.p>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: introStep >= 7 ? 1 : 0, y: introStep >= 7 ? 0 : 20 }}
              transition={{ duration: 0.8 }}
              className="flex flex-wrap items-center justify-center lg:justify-start gap-4 pt-2"
            >
              <Link to="/login">
                <Button
                  size="lg"
                  variant="primary"
                  rightIcon={<ArrowRight className="w-4 h-4" />}
                  className="shadow-[0_0_25px_rgba(0,229,255,0.4)] text-sm px-6 py-3"
                >
                  Enter Neural Lab
                </Button>
              </Link>

              <a href="#brain-explorer">
                <Button
                  size="lg"
                  variant="secondary"
                  leftIcon={<Play className="w-3.5 h-3.5 fill-current" />}
                  className="text-sm px-5 py-3 border-slate-700 hover:border-cyan-400"
                >
                  Explore the Technology
                </Button>
              </a>
            </motion.div>

            {/* Bottom 4 Feature Badges in a Row */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: introStep >= 7 ? 1 : 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="grid grid-cols-2 gap-2.5 pt-4 border-t border-slate-800/80 text-[11px] font-mono text-slate-300"
            >
              <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-900/60 border border-slate-800">
                <Brain className="w-4 h-4 text-cyan-400 shrink-0" />
                <span className="truncate">Advanced Deep Learning</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-900/60 border border-slate-800">
                <Atom className="w-4 h-4 text-violet-400 shrink-0" />
                <span className="truncate">Multi-Model Ensemble</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-900/60 border border-slate-800">
                <Activity className="w-4 h-4 text-pink-400 shrink-0" />
                <span className="truncate">EEG Signal Analysis</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-900/60 border border-slate-800">
                <Target className="w-4 h-4 text-emerald-400 shrink-0" />
                <span className="truncate">ADHD vs Control</span>
              </div>
            </motion.div>
          </div>

          {/* ======================================================= */}
          {/* COLUMN 2 (CENTER, ~46%): 3D BRAIN WITH LABELS & SIGNALS */}
          {/* ======================================================= */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: introStep >= 3 ? 1 : 0, scale: introStep >= 3 ? 1 : 0.9 }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            className="lg:col-span-5 xl:col-span-6 h-[550px] sm:h-[620px] md:h-[680px] relative flex flex-col justify-center items-center order-2"
          >
            {/* The 3D Human Brain WebGL Canvas with holographic pedestal */}
            <BrainCanvas
              activeRegion={activeHeroLobe}
              isOpeningCycleActive={true}
              interactive={true}
              onRegionClick={(k) => setActiveHeroLobe((prev) => (prev === k ? null : k))}
            />

            {/* SVG Glowing Pointer Lines to Brain Anatomical Regions */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none z-10 hidden sm:block">
              {/* Frontal Pointer: from Top-Left button to Anterior Cortex */}
              <line x1="28%" y1="12%" x2="40%" y2="28%" stroke="#00E5FF" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.75" />
              <circle cx="40%" cy="28%" r="4" fill="#00E5FF" className="animate-ping" />
              <circle cx="40%" cy="28%" r="2.5" fill="#FFFFFF" />

              {/* Parietal Pointer: from Top-Right button to Superior Parietal Cortex */}
              <line x1="72%" y1="12%" x2="60%" y2="26%" stroke="#8B5CF6" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.75" />
              <circle cx="60%" cy="26%" r="4" fill="#8B5CF6" className="animate-ping" />
              <circle cx="60%" cy="26%" r="2.5" fill="#FFFFFF" />

              {/* Temporal Pointer: from Bottom-Left button to Lateral Temporal Cortex */}
              <line x1="28%" y1="78%" x2="43%" y2="56%" stroke="#F02BD5" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.75" />
              <circle cx="43%" cy="56%" r="4" fill="#F02BD5" className="animate-ping" />
              <circle cx="43%" cy="56%" r="2.5" fill="#FFFFFF" />

              {/* Occipital Pointer: from Bottom-Right button to Posterior Occipital Cortex */}
              <line x1="72%" y1="78%" x2="64%" y2="52%" stroke="#35F5C8" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.75" />
              <circle cx="64%" cy="52%" r="4" fill="#35F5C8" className="animate-ping" />
              <circle cx="64%" cy="52%" r="2.5" fill="#FFFFFF" />
            </svg>

            {/* CALLOUT 1: TOP LEFT — FRONTAL LOBE */}
            <div className="absolute top-6 left-2 sm:left-4 z-20 pointer-events-auto">
              <button
                onClick={() => setActiveHeroLobe(activeHeroLobe === 'frontal' ? null : 'frontal')}
                className={`p-3 rounded-2xl border backdrop-blur-xl text-left transition-all ${
                  activeHeroLobe === 'frontal'
                    ? 'bg-slate-900/95 border-cyan-400 shadow-[0_0_25px_rgba(0,229,255,0.5)] scale-105'
                    : 'bg-slate-950/80 border-cyan-500/50 hover:border-cyan-400 hover:shadow-[0_0_20px_rgba(0,229,255,0.35)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#00e5ff]" />
                  <span className="text-xs sm:text-sm font-bold font-heading text-white">Frontal Lobe</span>
                </div>
                <span className="text-[10px] sm:text-xs font-mono font-bold text-cyan-300 block mt-0.5">
                  (Attention & Planning)
                </span>
              </button>
            </div>

            {/* CALLOUT 2: TOP RIGHT — PARIETAL LOBE */}
            <div className="absolute top-6 right-2 sm:right-4 z-20 pointer-events-auto">
              <button
                onClick={() => setActiveHeroLobe(activeHeroLobe === 'parietal' ? null : 'parietal')}
                className={`p-3 rounded-2xl border backdrop-blur-xl text-left transition-all ${
                  activeHeroLobe === 'parietal'
                    ? 'bg-slate-900/95 border-violet-400 shadow-[0_0_25px_rgba(139,92,246,0.5)] scale-105'
                    : 'bg-slate-950/80 border-violet-500/50 hover:border-violet-400 hover:shadow-[0_0_20px_rgba(139,92,246,0.35)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-violet-400 animate-pulse shadow-[0_0_8px_#8b5cf6]" />
                  <span className="text-xs sm:text-sm font-bold font-heading text-white">Parietal Lobe</span>
                </div>
                <span className="text-[10px] sm:text-xs font-mono font-bold text-violet-300 block mt-0.5">
                  (Sensory Processing)
                </span>
              </button>
            </div>

            {/* CALLOUT 3: BOTTOM LEFT — TEMPORAL LOBE */}
            <div className="absolute bottom-16 left-2 sm:left-4 z-20 pointer-events-auto">
              <button
                onClick={() => setActiveHeroLobe(activeHeroLobe === 'temporal' ? null : 'temporal')}
                className={`p-3 rounded-2xl border backdrop-blur-xl text-left transition-all ${
                  activeHeroLobe === 'temporal'
                    ? 'bg-slate-900/95 border-pink-400 shadow-[0_0_25px_rgba(240,43,213,0.5)] scale-105'
                    : 'bg-slate-950/80 border-pink-500/50 hover:border-pink-400 hover:shadow-[0_0_20px_rgba(240,43,213,0.35)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-pink-400 animate-pulse shadow-[0_0_8px_#f02bd5]" />
                  <span className="text-xs sm:text-sm font-bold font-heading text-white">Temporal Lobe</span>
                </div>
                <span className="text-[10px] sm:text-xs font-mono font-bold text-pink-300 block mt-0.5">
                  (Memory & Language)
                </span>
              </button>
            </div>

            {/* CALLOUT 4: BOTTOM RIGHT — OCCIPITAL LOBE */}
            <div className="absolute bottom-16 right-2 sm:right-4 z-20 pointer-events-auto">
              <button
                onClick={() => setActiveHeroLobe(activeHeroLobe === 'occipital' ? null : 'occipital')}
                className={`p-3 rounded-2xl border backdrop-blur-xl text-left transition-all ${
                  activeHeroLobe === 'occipital'
                    ? 'bg-slate-900/95 border-teal-400 shadow-[0_0_25px_rgba(53,245,200,0.5)] scale-105'
                    : 'bg-slate-950/80 border-teal-500/50 hover:border-teal-400 hover:shadow-[0_0_20px_rgba(53,245,200,0.35)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-teal-400 animate-pulse shadow-[0_0_8px_#35f5c8]" />
                  <span className="text-xs sm:text-sm font-bold font-heading text-white">Occipital Lobe</span>
                </div>
                <span className="text-[10px] sm:text-xs font-mono font-bold text-teal-300 block mt-0.5">
                  (Visual Processing)
                </span>
              </button>
            </div>

            {/* FLOATING MINI WIDGET: EEG SIGNALS (Left of Brain) */}
            <div className="absolute left-2 sm:left-4 top-1/2 -translate-y-1/2 z-20 p-2.5 rounded-2xl glass-panel border border-cyan-500/30 backdrop-blur-xl shadow-2xl space-y-1.5 hidden md:block max-w-[130px]">
              <div className="flex items-center justify-between pb-1 border-b border-slate-800">
                <span className="text-[9px] font-mono font-bold text-cyan-300">EEG Signals</span>
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
              </div>

              {[
                { ch: 'Fp1', col: '#00E5FF' },
                { ch: 'Fp2', col: '#F02BD5' },
                { ch: 'F7', col: '#35F5C8' },
                { ch: 'F8', col: '#287BFF' },
                { ch: 'T3', col: '#8B5CF6' },
                { ch: 'T4', col: '#35F5C8' },
                { ch: 'O1', col: '#00E5FF' },
                { ch: 'O2', col: '#F02BD5' },
              ].map((sig, idx) => (
                <div key={idx} className="flex items-center gap-1">
                  <span className="text-[8px] font-mono text-slate-400 w-4">{sig.ch}</span>
                  <svg viewBox="0 0 50 8" className="w-16 h-2" preserveAspectRatio="none">
                    <path
                      d="M0,4 Q5,1 10,4 T20,4 T30,4 T40,4 T50,4"
                      fill="none"
                      stroke={sig.col}
                      strokeWidth="1.2"
                    />
                  </svg>
                </div>
              ))}
            </div>

            {/* Bottom Mouse Scroll to Explore Indicator */}
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-[10px] font-mono text-slate-400 pointer-events-none">
              <span className="w-4 h-6 rounded-full border border-slate-600 flex items-start justify-center p-1">
                <span className="w-1 h-1.5 rounded-full bg-cyan-400 animate-bounce" />
              </span>
              <span>Scroll to explore</span>
              <ChevronDown className="w-3.5 h-3.5 text-cyan-400 -mt-1 animate-pulse" />
            </div>

            {/* Lobe Educational Details Popover (when a lobe is selected) */}
            <AnimatePresence>
              {activeHeroLobe && (
                <motion.div
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute bottom-16 left-6 right-6 z-30 p-4 rounded-2xl glass-panel border border-slate-700 backdrop-blur-2xl shadow-2xl space-y-2 text-left"
                >
                  {(() => {
                    const lobe = BRAIN_REGIONS.find((r) => r.key === activeHeroLobe);
                    if (!lobe) return null;
                    return (
                      <>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: lobe.color }}
                            />
                            <h4 className="text-sm font-bold font-heading text-white">
                              {lobe.name} ({lobe.scientificName})
                            </h4>
                          </div>
                          <button
                            onClick={() => setActiveHeroLobe(null)}
                            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed font-sans">
                          {lobe.functionDescription}
                        </p>
                        <div className="flex items-center gap-2 text-[10px] font-mono text-cyan-300 pt-1 border-t border-slate-800">
                          <span className="text-slate-400">10-20 Electrodes:</span>
                          <span className="font-bold">{lobe.channels.join(', ')}</span>
                        </div>
                      </>
                    );
                  })()}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* ======================================================= */}
          {/* COLUMN 3 (RIGHT, ~26%): UNDERSTAND ADHD MULTI-PANEL */}
          {/* ======================================================= */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: introStep >= 6 ? 1 : 0, x: introStep >= 6 ? 0 : 20 }}
            transition={{ duration: 0.8 }}
            className="lg:col-span-3 xl:col-span-3 order-3"
          >
            <UnderstandAdhdPanel />
          </motion.div>
        </div>
      </section>

      {/* ========================================================= */}
      {/* 2. SIX BOTTOM FEATURE CARDS & SCI-FI STRIP (From Reference) */}
      {/* ========================================================= */}
      <FeatureCardsSection />

      {/* ========================================================= */}
      {/* 3. INTERACTIVE BRAIN EXPLORER (The Brain in Action) */}
      {/* ========================================================= */}
      <BrainExplorer />

      {/* ========================================================= */}
      {/* 4. DEDICATED 19-CHANNEL 10-20 EEG SIGNALS SECTION */}
      {/* ========================================================= */}
      <EegChannelsSection />

      {/* ========================================================= */}
      {/* 5. THE 3 AI MODELS SECTION (3D Tilt & Canvas Feature Maps) */}
      {/* ========================================================= */}
      <ModelsSection />

      {/* ========================================================= */}
      {/* 6. HOW THE PREDICTION WORKS PIPELINE (From EEG to Prediction) */}
      {/* ========================================================= */}
      <PredictionWorkflowSection />

      {/* ========================================================= */}
      {/* 7. BOTTOM CALL TO ACTION BANNER */}
      {/* ========================================================= */}
      <section className="py-20 px-4 md:px-8 max-w-5xl mx-auto w-full text-center">
        <div className="p-10 md:p-16 rounded-3xl glass-panel border border-cyan-500/30 relative overflow-hidden space-y-6 shadow-[0_0_50px_rgba(0,229,255,0.15)]">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-violet-600/20 border border-cyan-400/40 flex items-center justify-center mx-auto shadow-xl">
            <Brain className="w-8 h-8 text-cyan-400" />
          </div>

          <div className="space-y-2">
            <h2 className="text-3xl md:text-4xl font-bold font-heading text-white">
              Ready to Classify EEG Recordings?
            </h2>
            <p className="text-slate-400 text-sm md:text-base max-w-lg mx-auto">
              Launch the live Neural Lab to upload raw EEG files or ZIP archives and inspect calibrated predictions from our multi-model ensemble.
            </p>
          </div>

          <div className="pt-2 flex flex-wrap items-center justify-center gap-4">
            <Link to="/login">
              <Button size="lg" variant="primary" rightIcon={<ArrowRight className="w-5 h-5" />}>
                Launch Neural Lab Workspace
              </Button>
            </Link>
            <Link to="/dashboard/upload">
              <Button size="lg" variant="secondary">
                Upload EEG File Now
              </Button>
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};
