import { useState, useEffect } from 'react';

export default function BirthdayPartyOptimization() {
  // États pour les valeurs des curseurs (entre 0 et 1)
  const [amisVal, setAmisVal] = useState(0.56);
  const [jeuxVal, setJeuxVal] = useState(0.56);
  const [snacksVal, setSnacksVal] = useState(0.44);
  const [budgetVal, setBudgetVal] = useState(0.69);
  
  // États pour les valeurs du radar
  const [amisScore, setAmisScore] = useState(0);
  const [jeuxScore, setJeuxScore] = useState(0);
  const [snacksScore, setSnacksScore] = useState(0);
  const [budgetScore, setBudgetScore] = useState(0);
  
  // Mise à jour des scores du radar en fonction des valeurs des curseurs
  useEffect(() => {
    // Calcul direct des scores pour amis, jeux et snacks
    const amisScoreNew = amisVal * 180;
    const jeuxScoreNew = jeuxVal * 180;
    const snacksScoreNew = snacksVal * 180;
    
    // Le budget est influencé par les autres paramètres
    const budgetImpact = 1.0 - ((amisVal * 0.4) + (jeuxVal * 0.3) + (snacksVal * 0.3));
    let budgetScoreNew = budgetVal * budgetImpact * 180;
    if (budgetScoreNew < 20) budgetScoreNew = 20; // Score minimum pour le budget
    
    setAmisScore(amisScoreNew);
    setJeuxScore(jeuxScoreNew);
    setSnacksScore(snacksScoreNew);
    setBudgetScore(budgetScoreNew);
  }, [amisVal, jeuxVal, snacksVal, budgetVal]);
  
  // Gestion des déplacements des curseurs
  const handleSliderChange = (setter) => (e) => {
    const trackWidth = 160;
    const value = Math.max(0, Math.min(1, (parseInt(e.target.value) / 100)));
    setter(value);
  };
  
  return (
    <div className="flex flex-col items-center w-full">
      <svg viewBox="0 0 800 600" className="w-full max-w-4xl">
        {/* Définition des styles */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#ff6347" />
          </marker>
          <marker id="positive-arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#4caf50" />
          </marker>
          <marker id="negative-arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#f44336" />
          </marker>
        </defs>

        {/* Titre et sous-titre */}
        <text x="400" y="40" textAnchor="middle" fontSize="20px" fontWeight="bold" fontFamily="Arial, sans-serif">
          Optimisation Multi-objectif : Fête d'Anniversaire
        </text>
        <text x="400" y="65" textAnchor="middle" fontSize="16px" fontStyle="italic" fontFamily="Arial, sans-serif">
          Équilibrer 4 objectifs en conflit
        </text>

        {/* Graphique radar */}
        <g transform="translate(400, 300)">
          {/* Cercles de référence */}
          <circle cx="0" cy="0" r="50" stroke="#ccc" strokeWidth="1" strokeDasharray="5,5" fill="none" />
          <circle cx="0" cy="0" r="100" stroke="#ccc" strokeWidth="1" strokeDasharray="5,5" fill="none" />
          <circle cx="0" cy="0" r="150" stroke="#ccc" strokeWidth="1" strokeDasharray="5,5" fill="none" />
          <circle cx="0" cy="0" r="200" stroke="#ccc" strokeWidth="1" strokeDasharray="5,5" fill="none" />
          
          {/* Axes */}
          <line x1="0" y1="0" x2="0" y2="-200" stroke="#999" strokeWidth="1" />
          <line x1="0" y1="0" x2="200" y2="0" stroke="#999" strokeWidth="1" />
          <line x1="0" y1="0" x2="0" y2="200" stroke="#999" strokeWidth="1" />
          <line x1="0" y1="0" x2="-200" y2="0" stroke="#999" strokeWidth="1" />
          
          {/* Étiquettes des axes */}
          <text x="0" y="-210" textAnchor="middle" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Amis</text>
          <text x="210" y="0" textAnchor="start" dominantBaseline="middle" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Jeux</text>
          <text x="0" y="220" textAnchor="middle" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Snacks</text>
          <text x="-210" y="0" textAnchor="end" dominantBaseline="middle" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Budget (économies)</text>
          
          {/* Polygon du radar */}
          <polygon 
            points={`0,${-amisScore} ${jeuxScore},0 0,${snacksScore} ${-budgetScore},0`} 
            fill="rgba(65, 105, 225, 0.5)" 
            stroke="#4169e1" 
            strokeWidth="2"
          />
          
          {/* Points sur le radar */}
          <circle cx="0" cy={-amisScore} r="6" fill="#ff6347" stroke="#c83a2e" strokeWidth="2" />
          <circle cx={jeuxScore} cy="0" r="6" fill="#ff6347" stroke="#c83a2e" strokeWidth="2" />
          <circle cx="0" cy={snacksScore} r="6" fill="#ff6347" stroke="#c83a2e" strokeWidth="2" />
          <circle cx={-budgetScore} cy="0" r="6" fill="#ff6347" stroke="#c83a2e" strokeWidth="2" />
        </g>

        {/* Légende explicative */}
        <rect x="550" y="100" width="220" height="150" fill="#f9f9f9" stroke="#ccc" strokeWidth="1" rx="5" ry="5" />
        <text x="560" y="125" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Comment utiliser:</text>
        <text x="560" y="150" fontSize="14px" fontFamily="Arial, sans-serif">1. Déplacez les curseurs pour ajuster</text>
        <text x="560" y="170" fontSize="14px" fontFamily="Arial, sans-serif">   les paramètres de la fête</text>
        <text x="560" y="190" fontSize="14px" fontFamily="Arial, sans-serif">2. Observez l'impact sur les autres</text>
        <text x="560" y="210" fontSize="14px" fontFamily="Arial, sans-serif">   objectifs dans le graphique radar</text>
        <text x="560" y="230" fontSize="14px" fontFamily="Arial, sans-serif">3. Cherchez le meilleur équilibre!</text>

        {/* Explication des interactions */}
        <rect x="100" y="100" width="220" height="150" fill="#f9f9f9" stroke="#ccc" strokeWidth="1" rx="5" ry="5" />
        <text x="110" y="125" fontSize="14px" fontWeight="bold" fontFamily="Arial, sans-serif">Interactions entre objectifs:</text>
        <text x="110" y="150" fontSize="14px" fontFamily="Arial, sans-serif">• Plus d'amis = Plus de coûts</text>
        <text x="110" y="170" fontSize="14px" fontFamily="Arial, sans-serif">• Meilleurs jeux = Plus de coûts</text>
        <text x="110" y="190" fontSize="14px" fontFamily="Arial, sans-serif">• Plus de snacks = Plus de coûts</text>
        <text x="110" y="210" fontSize="14px" fontFamily="Arial, sans-serif">• Budget limité = Compromis nécessaires</text>
        <text x="110" y="230" fontSize="14px" fontFamily="Arial, sans-serif">  sur les autres objectifs</text>
      </svg>

      {/* Sliders interactifs */}
      <div className="w-full max-w-4xl mt-4 px-4">
        {/* Slider pour les amis */}
        <div className="mb-6">
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium">Nombre d'amis:</label>
            <span className="text-sm">{Math.round(amisVal * 100)}%</span>
          </div>
          <div className="relative">
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={amisVal * 100} 
              onChange={handleSliderChange(setAmisVal)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="mt-2 flex justify-between text-xs">
              <span className="text-green-600">Plus d'amis ➜ Plus de plaisir!</span>
              <span className="text-red-600">Mais aussi ➜ Plus de coûts</span>
            </div>
          </div>
        </div>

        {/* Slider pour les jeux */}
        <div className="mb-6">
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium">Qualité des jeux:</label>
            <span className="text-sm">{Math.round(jeuxVal * 100)}%</span>
          </div>
          <div className="relative">
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={jeuxVal * 100} 
              onChange={handleSliderChange(setJeuxVal)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="mt-2 flex justify-between text-xs">
              <span className="text-green-600">Meilleurs jeux ➜ Plus de fun!</span>
              <span className="text-red-600">Mais aussi ➜ Plus de coûts</span>
            </div>
          </div>
        </div>

        {/* Slider pour les snacks */}
        <div className="mb-6">
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium">Variété des snacks:</label>
            <span className="text-sm">{Math.round(snacksVal * 100)}%</span>
          </div>
          <div className="relative">
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={snacksVal * 100} 
              onChange={handleSliderChange(setSnacksVal)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="mt-2 flex justify-between text-xs">
              <span className="text-green-600">Plus de snacks ➜ Plus savoureux!</span>
              <span className="text-red-600">Mais aussi ➜ Plus de coûts</span>
            </div>
          </div>
        </div>

        {/* Slider pour le budget */}
        <div className="mb-6">
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium">Budget max:</label>
            <span className="text-sm">{Math.round(budgetVal * 100)}%</span>
          </div>
          <div className="relative">
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={budgetVal * 100} 
              onChange={handleSliderChange(setBudgetVal)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="mt-2 flex justify-between text-xs">
              <span className="text-green-600">Budget élevé ➜ Plus d'options!</span>
              <span className="text-red-600">Mais moins d'économies</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
