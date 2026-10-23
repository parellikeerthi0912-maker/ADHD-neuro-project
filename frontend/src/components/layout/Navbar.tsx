import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Cpu, Sparkles, Layers, ShieldCheck, ArrowRight } from 'lucide-react';
import { getHealth } from '../../services/api';
import { HealthResponse } from '../../types';

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    getHealth()
      .then((data) => {
        if (isMounted) setHealth(data);
      })
      .catch(() => {
        if (isMounted) setHealth(null);
      })
      .finally(() => {
        if (isMounted) setIsChecking(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-[#050816]/75 backdrop-blur-xl border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 md:px-8 h-20 flex items-center justify-between">
        {/* Brand Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-violet-600/20 border border-cyan-400/40 flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.25)] group-hover:border-cyan-300 transition-all">
            <Activity className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
            <div className="absolute inset-0 rounded-xl bg-cyan-400/10 filter blur-sm group-hover:opacity-100 transition-opacity" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-lg md:text-xl text-white tracking-wider">
                NEUROSCAN <span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-300">
                v3.0
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono tracking-tight hidden sm:block">
              EEG-Based ADHD Detection
            </p>
          </div>
        </Link>

        {/* Center Nav Links */}
        <nav className="hidden lg:flex items-center gap-7 text-xs font-mono font-medium text-slate-300">
          <Link to="/" className="text-white border-b-2 border-cyan-400 py-1 font-semibold">
            Home
          </Link>
          <a href="/#understand-adhd" className="hover:text-cyan-400 transition-colors py-1">
            About
          </a>
          <a href="/#models-section" className="hover:text-cyan-400 transition-colors py-1">
            Models
          </a>
          <Link to="/dashboard/results" className="hover:text-cyan-400 transition-colors py-1">
            Results
          </Link>
          <a href="#footer" className="hover:text-cyan-400 transition-colors py-1">
            Contact
          </a>
        </nav>

        {/* Right Action & Backend Status */}
        <div className="flex items-center gap-3">
          {/* Backend Connection Indicator */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-800 text-xs font-mono">
            {isChecking ? (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-slate-400">Pinging API</span>
              </>
            ) : health ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10B981]" />
                <span className="text-emerald-400 font-medium">FastAPI Connected</span>
                <span className="text-[10px] text-slate-500">({health.device})</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                <span className="text-rose-400">API Offline</span>
              </>
            )}
          </div>

          {/* Login Pill Button with User Icon matching reference */}
          {location.pathname !== '/login' && (
            <Link
              to="/login"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-mono font-semibold bg-slate-900/90 border border-cyan-400/50 text-white shadow-[0_0_15px_rgba(0,229,255,0.25)] hover:border-cyan-300 hover:shadow-[0_0_25px_rgba(0,229,255,0.45)] hover:scale-105 transition-all"
            >
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span>Login</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};
