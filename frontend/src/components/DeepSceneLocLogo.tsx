import React from "react";

export default function DeepSceneLocLogo({ className = "w-10 h-8" }: { className?: string }) {
  return (
    <svg 
      id="deepseneloc-logo-svg"
      viewBox="0 0 512 360" 
      className={`${className} transition-all duration-300 hover:scale-[1.03]`}
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        {/* Metallic body gradient */}
        <linearGradient id="metal-body" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#e2e8f0" />
          <stop offset="30%" stopColor="#cbd5e1" />
          <stop offset="50%" stopColor="#94a3b8" />
          <stop offset="70%" stopColor="#cbd5e1" />
          <stop offset="100%" stopColor="#64748b" />
        </linearGradient>

        {/* Shutter & dials metal gradient */}
        <linearGradient id="dial-grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#94a3b8" />
          <stop offset="50%" stopColor="#475569" />
          <stop offset="100%" stopColor="#1e293b" />
        </linearGradient>

        {/* Deep blue letter textures */}
        <linearGradient id="dl-letters" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#1d4ed8" />
          <stop offset="50%" stopColor="#1e40af" />
          <stop offset="100%" stopColor="#172554" />
        </linearGradient>

        {/* Golden aperture ring */}
        <linearGradient id="gold-aperture" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#fbbf24" />
          <stop offset="50%" stopColor="#d97706" />
          <stop offset="100%" stopColor="#78350f" />
        </linearGradient>

        {/* Glass lens reflection overlay */}
        <linearGradient id="glass-glare" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.4" />
          <stop offset="40%" stopColor="#ffffff" stopOpacity="0.05" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
        </linearGradient>

        {/* Segment Gradients */}
        {/* 1. Urban (Indigo/Cyan) */}
        <linearGradient id="urban-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#312e81" />
        </linearGradient>

        {/* 2. Forest (Emerald/Grass) */}
        <linearGradient id="forest-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#064e3b" />
        </linearGradient>

        {/* 3. Mountain (Purple/Charcoal) */}
        <linearGradient id="mountain-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#4c1d95" />
        </linearGradient>

        {/* 4. Coastal (Teal/Cyan) */}
        <linearGradient id="coastal-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#164e63" />
        </linearGradient>

        {/* 5. Rural (Amber/Yellow) */}
        <linearGradient id="rural-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#78350f" />
        </linearGradient>

        {/* Textpath for curved DeepSceneLoc labeling */}
        <path id="text-arc-top" d="M 180,140 A 88 88 0 0 1 332,140" />
      </defs>

      {/* 1. Camera Body External Hardware */}
      {/* Top hot shoe & viewfinder bump */}
      <path 
        d="M 190,52 L 210,18 L 302,18 L 322,52 Z" 
        fill="url(#metal-body)" 
        stroke="#1e293b" 
        strokeWidth="6" 
        strokeLinejoin="round" 
      />
      <rect x="220" y="24" width="72" height="28" rx="4" fill="#0f172a" stroke="#475569" strokeWidth="2" />
      <line x1="256" y1="24" x2="256" y2="52" stroke="#64748b" strokeWidth="2" />

      {/* Left Shutter dial button */}
      <rect x="76" y="22" width="64" height="30" rx="6" fill="url(#dial-grad)" stroke="#1e293b" strokeWidth="5" />
      <line x1="88" y1="28" x2="128" y2="28" stroke="#cbd5e1" strokeWidth="2" />
      <line x1="88" y1="34" x2="128" y2="34" stroke="#cbd5e1" strokeWidth="2" />

      {/* Right control dial rotater */}
      <rect x="372" y="26" width="76" height="26" rx="4" fill="url(#dial-grad)" stroke="#1e293b" strokeWidth="5" />
      
      {/* Camera Body Main Plate */}
      <rect 
        x="12" 
        y="50" 
        width="488" 
        height="298" 
        rx="42" 
        fill="url(#metal-body)" 
        stroke="#1e293b" 
        strokeWidth="8" 
      />

      {/* Ivory/Silver Faceplate Insert (Highly detailed scientific background) */}
      <rect 
        x="24" 
        y="62" 
        width="464" 
        height="274" 
        rx="32" 
        fill="#f8fafc" 
        stroke="#cbd5e1" 
        strokeWidth="3" 
      />

      {/* 2. Microchip Circuits Graphics (Deep Learning GIS aesthetic) */}
      {/* PCB circuit trace lines on left */}
      <path d="M 36,90 L 90,90 L 110,110 L 110,150" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
      <path d="M 36,110 L 76,110 L 86,120 L 86,160" stroke="#cbd5e1" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="110" cy="150" r="4" fill="#64748b" />
      <circle cx="86" cy="160" r="3" fill="#94a3b8" />

      {/* PCB circuit trace lines on right connected to a microchip */}
      <path d="M 476,210 L 416,210 L 396,230 L 396,270" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
      <circle cx="476" cy="210" r="4.5" fill="#3b82f6" />
      
      {/* Right corner "DSL Integration Socket" microchip */}
      <rect x="416" y="80" width="56" height="34" rx="4" fill="#1e293b" stroke="#475569" strokeWidth="2" />
      <text x="444" y="102" fill="#38bdf8" fontSize="13" fontWeight="bold" fontFamily="monospace" textAnchor="middle">DSL</text>
      {/* Solder pins */}
      <line x1="424" y1="74" x2="424" y2="80" stroke="#94a3b8" strokeWidth="2" />
      <line x1="434" y1="74" x2="434" y2="80" stroke="#94a3b8" strokeWidth="2" />
      <line x1="444" y1="74" x2="444" y2="80" stroke="#94a3b8" strokeWidth="2" />
      <line x1="454" y1="74" x2="454" y2="80" stroke="#94a3b8" strokeWidth="2" />

      {/* Mini Aperture calibration icons (Bottom Left) */}
      <circle cx="50" cy="296" r="16" stroke="#475569" strokeWidth="1.5" />
      <path d="M 50,280 L 46,290 M 50,312 L 54,302 M 34,296 L 44,298 M 66,296 L 56,294" stroke="#475569" strokeWidth="1" />
      
      <circle cx="90" cy="296" r="16" stroke="#475569" strokeWidth="1.5" />
      <path d="M 90,280 L 86,292 M 90,312 L 94,300" stroke="#475569" strokeWidth="1" />

      <circle cx="130" cy="296" r="16" stroke="#475569" strokeWidth="1.5" />

      {/* ISO / Metadata Info Box (Header margin style) */}
      <text x="350" y="74" fill="#64748b" fontSize="9.5" fontFamily="monospace" fontWeight="bold">ISO: AUTO | F-STOP: 2.8 | MODE: AUTO</text>

      {/* 3. Deep integrated letters "D", "S", "L" */}
      {/* Giant "D" - left side */}
      <path 
        d="M 28,110 L 90,110 C 130,110 156,134 156,176 C 156,218 130,242 90,242 L 28,242 Z 
           M 58,138 L 58,214 L 88,214 C 114,214 126,202 126,176 C 126,150 114,138 88,138 Z" 
        fill="url(#dl-letters)" 
        stroke="#0f172a" 
        strokeWidth="3.5"
      />

      {/* Giant "S" - styled looping above the lens center to form the crest */}
      <path 
        d="M 194,116 C 194,84 218,62 256,62 C 294,62 318,84 318,114 L 288,114 C 288,100 274,90 256,90 C 238,90 224,100 224,112 C 224,142 318,132 318,178 C 318,214 292,238 256,238 C 216,238 190,212 190,178 L 220,178 C 220,198 236,210 256,210 C 276,210 288,198 288,184 C 288,154 194,162 194,116 Z" 
        fill="url(#dl-letters)" 
        stroke="#0f172a" 
        strokeWidth="3.5"
      />

      {/* Giant "L" - right side */}
      <path 
        d="M 374,110 L 404,110 L 404,214 L 452,214 L 452,242 L 374,242 Z" 
        fill="url(#dl-letters)" 
        stroke="#0f172a" 
        strokeWidth="3.5"
      />

      {/* Small mini-aperture inside "S" top loop */}
      <circle cx="256" cy="113" r="14" fill="#0f172a" stroke="#e2e8f0" strokeWidth="2" />
      <polygon points="256,101 266,113 250,121" fill="#475569" />
      <polygon points="244,113 256,125 258,107" fill="#64748b" />

      {/* 4. Central Large Lens Assembly */}
      {/* Outer black grip collar ring */}
      <circle cx="256" cy="198" r="110" fill="#0f172a" stroke="#334155" strokeWidth="6" />
      
      {/* Textured grip ridges on lens ring */}
      <circle cx="256" cy="198" r="105" fill="none" stroke="#1e293b" strokeWidth="3" strokeDasharray="6 3" />

      {/* Golden Aperture Dial Ring */}
      <circle cx="256" cy="198" r="98" fill="none" stroke="url(#gold-aperture)" strokeWidth="6" />

      {/* Dark Inner Chamber */}
      <circle cx="256" cy="198" r="92" fill="#020617" />

      {/* Arched Text Branding: "DeepSceneLoc" */}
      <text fill="#ffffff" fontSize="13" fontWeight="900" fontFamily="sans-serif" letterSpacing="0.12em">
        {/* Curved path embedding fallback for SVG renderers */}
        <textPath href="#text-arc-top" startOffset="50%" textAnchor="middle">
          DEEPSCENELOC
        </textPath>
      </text>

      {/* ─── SCENE CLASSIFIER LENS SEGMENTS (Aperture wedges displaying domains) ─── */}
      <g clipPath="url(#lens-clip)">
        {/* We use a clip path to keep everything in the inner circular lens glass (R=76, Center=256,198) */}
        
        {/* 1. Urban (Top Wedge) - Skyline towers */}
        <path 
          d="M 256,198 L 256,122 A 76 76 0 0 1 328,172 Z" 
          fill="url(#urban-grad)" 
          stroke="#090d16" 
          strokeWidth="2.5" 
        />
        {/* Micro-skyline silhouette in Urban sector */}
        <rect x="274" y="148" width="12" height="22" fill="#e0e7ff" opacity="0.45" />
        <rect x="288" y="140" width="10" height="30" fill="#e0e7ff" opacity="0.6" />
        <polygon points="293,130 288,140 298,140" fill="#e0e7ff" opacity="0.6" />
        <rect x="302" y="152" width="14" height="15" fill="#e0e7ff" opacity="0.3" />

        {/* 2. Forest (Middle Right Wedge) - Pine tree shapes */}
        <path 
          d="M 256,198 L 328,172 A 76 76 0 0 1 300,268 Z" 
          fill="url(#forest-grad)" 
          stroke="#090d16" 
          strokeWidth="2.5" 
        />
        {/* Micro evergreen trees in Forest sector */}
        <polygon points="296,204 286,220 306,220" fill="#a7f3d0" opacity="0.7" />
        <polygon points="296,212 282,232 310,232" fill="#6ee7b7" opacity="0.85" />
        <polygon points="314,216 304,230 324,230" fill="#a7f3d0" opacity="0.5" />

        {/* 3. Mountain (Bottom Wedge) - Sharp alpine heights */}
        <path 
          d="M 256,198 L 300,268 A 76 76 0 0 1 212,268 Z" 
          fill="url(#mountain-grad)" 
          stroke="#090d16" 
          strokeWidth="2.5" 
        />
        {/* Peak vectors in Mountain sector */}
        <polygon points="256,230 236,268 276,268" fill="#ddd6fe" opacity="0.5" />
        {/* Snowy cap overlay */}
        <polygon points="256,230 250,242 262,242" fill="#ffffff" />
        
        <polygon points="274,242 260,268 288,268" fill="#c084fc" opacity="0.6" />
        <polygon points="274,242 270,250 278,250" fill="#ffffff" />

        {/* 4. Coastal (Middle Left Wedge) - Wave lines & tiny boat */}
        <path 
          d="M 256,198 L 212,268 A 76 76 0 0 1 184,172 Z" 
          fill="url(#coastal-grad)" 
          stroke="#090d16" 
          strokeWidth="2.5" 
        />
        {/* Ocean ripples and yacht in Coastal sector */}
        <path d="M 194,220 C 204,220 200,224 210,224 C 220,224 216,220 226,220" stroke="#ecfeff" strokeWidth="1.5" strokeLinecap="round" opacity="0.7" />
        <path d="M 190,234 C 200,234 196,238 206,238 C 216,238 212,234 222,234" stroke="#ecfeff" strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
        {/* Triangle sailboat */}
        <polygon points="208,206 200,214 208,214" fill="#ffffff" />
        <line x1="198" y1="215" x2="210" y2="215" stroke="#ffffff" strokeWidth="1" />

        {/* 5. Rural (Top Left Wedge) - Farmland furrows */}
        <path 
          d="M 256,198 L 184,172 A 76 76 0 0 1 256,122 Z" 
          fill="url(#rural-grad)" 
          stroke="#090d16" 
          strokeWidth="2.5" 
        />
        {/* Converging soil furrow lines in Rural sector */}
        <line x1="256" y1="198" x2="204" y2="140" stroke="#fef08a" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.8" />
        <line x1="256" y1="198" x2="222" y2="132" stroke="#fef08a" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.8" />
        <line x1="256" y1="198" x2="242" y2="126" stroke="#fef08a" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.8" />
        {/* Micro sun flare */}
        <circle cx="204" cy="144" r="5" fill="#fef08a" />
      </g>

      {/* Clip path definition for the standard circular aperture mask */}
      <clipPath id="lens-clip">
        <circle cx="256" cy="198" r="76" />
      </clipPath>

      {/* Central Aperture Shutter Blades & Reflection Overlays (Creates camera lens realistic gloss) */}
      <circle cx="256" cy="198" r="22" fill="#020617" stroke="url(#gold-aperture)" strokeWidth="3" />
      {/* Blades radiating outwards from center */}
      <path d="M 256,176 L 268,198 M 278,198 L 256,209 M 256,220 L 244,198 M 234,198 L 256,187" stroke="#1e293b" strokeWidth="2" />
      <circle cx="256" cy="198" r="8" fill="#000000" />

      {/* Front Reflection Highlight Glare (Highly realistic glass sphere curvature) */}
      <circle cx="256" cy="198" r="76" fill="url(#glass-glare)" pointerEvents="none" />
      <path d="M 194,154 A 76 76 0 0 1 234,126" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" fill="none" opacity="0.32" pointerEvents="none" />
      <circle cx="310" cy="160" r="3" fill="#ffffff" opacity="0.4" pointerEvents="none" />
    </svg>
  );
}
