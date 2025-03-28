import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Label, Sector } from 'recharts';

// Updated colorblind-friendly palette
// Using a palette designed for all types of color blindness
// Blue, orange, purple, teal, gold, etc. with good contrast
const COLORS = {
  'credit_score': '#3498db',       // Blue
  'DTI': '#9b59b6',               // Purple
  'payment_history': '#1abc9c',    // Teal
  'estimated_debt': '#f39c12',     // Orange/Gold
  'months_employed': '#e67e22',    // Dark Orange
  'others': '#95a5a6',             // Gray
  'empty': '#ecf0f1',              // Light Gray
  'potential': '#c0392b',          // Dark Red
  'approval': '#27ae60',           // Dark Green
  'rejection': '#ecf0f1',          // Light Gray
  'threshold': '#d35400'           // Burnt Orange
};

// Updated thresholds with descriptions and current value comparisons
const THRESHOLDS = {
  'credit_score': {
    target: 660,
    description: 'Minimum credit score for better approval odds',
    format: (value) => value,
    compare: (current, target) => current >= target
  },
  'DTI': {
    target: 40,
    description: 'Maximum debt-to-income ratio recommended',
    format: (value) => value + '%',
    compare: (current, target) => current <= target
  },
  'payment_history': {
    target: 0.9,
    description: 'Minimum on-time payment ratio',
    format: (value) => (value * 100) + '%',
    compare: (current, target) => current >= target
  },
  'estimated_debt': {
    target: 0.4,
    description: 'Maximum recommended debt ratio',
    format: (value) => (value * 100) + '%',
    compare: (current, target) => current <= target
  },
  'months_employed': {
    target: 12,
    description: 'Minimum months of employment preferred',
    format: (value) => value + ' months',
    compare: (current, target) => current >= target
  }
};

// Extract base feature name without qualifiers
const getBaseFeature = (feature) => {
  if (feature.startsWith('payment_history_')) return 'payment_history';
  return feature;
};

const formatFeatureName = (name) => {
  // Remove prefixes and format the name
  let formattedName = name
    .replace(/(payment_history_|employment_status_|province_)/, '')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  
  // Special formatting for DTI
  if (name === 'DTI') return 'Debt-to-Income';
  return formattedName;
};

const CustomLabel = ({ viewBox, value, thresholdValue }) => {
  const { cx, cy } = viewBox;
  return (
    <g>
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle">
        <tspan x={cx} dy="-1em" fontSize="24" fontWeight="bold">
          {value} pts
        </tspan>
        <tspan x={cx} dy="1.5em" fontSize="14" fill="#666">
          Approval Score
        </tspan>
      </text>
    </g>
  );
};

// Arrow showing threshold or current position
const ArrowMarker = ({ cx, cy, outerRadius, angle, text, isThreshold, color = "#333" }) => {
  // Convert angle from degrees to radians
  const radian = (angle * Math.PI) / 180;
  
  // Calculate position - slightly outside the pie
  const x = cx + (outerRadius + 15) * Math.cos(radian);
  const y = cy + (outerRadius + 15) * Math.sin(radian);
  
  // Calculate arrow points
  const arrowSize = 8;
  const arrowX = cx + (outerRadius + 5) * Math.cos(radian);
  const arrowY = cy + (outerRadius + 5) * Math.sin(radian);
  
  return (
    <g>
      {/* Line */}
      <line
        x1={arrowX}
        y1={arrowY}
        x2={x}
        y2={y}
        stroke={color}
        strokeWidth={2}
      />
      
      {/* Arrow head */}
      <polygon
        points={`${arrowX},${arrowY} ${arrowX + arrowSize * Math.cos(radian - Math.PI/6)},${arrowY + arrowSize * Math.sin(radian - Math.PI/6)} ${arrowX + arrowSize * Math.cos(radian + Math.PI/6)},${arrowY + arrowSize * Math.sin(radian + Math.PI/6)}`}
        fill={color}
      />
      
      {/* Text */}
      <text
        x={x + 10 * Math.cos(radian)}
        y={y + 10 * Math.sin(radian)}
        fill={color}
        textAnchor={angle > 90 && angle < 270 ? "end" : "start"}
        dominantBaseline="middle"
        fontSize={12}
        fontWeight={isThreshold ? "normal" : "bold"}
      >
        {text}
      </text>
    </g>
  );
};

// Custom Active Shape component for highlighting sectors
const renderActiveShape = (props) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props;
  
  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 6}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        stroke="#333"
        strokeWidth={2}
      />
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
    </g>
  );
};

// Pie chart for showing factors (positive or negative)
const FactorPieChart = ({ data, title, type, approvalThreshold, totalApprovalPoints }) => {
  const [activeIndex, setActiveIndex] = React.useState(null);
  
  // Define colorblind-friendly color schemes
  const getAccessibleColor = (index, isPositive) => {
    // Colorblind-friendly palettes
    const positiveColors = [
      '#27ae60', // Dark Green
      '#2ecc71', // Green
      '#16a085', // Teal Green
      '#1abc9c', // Teal
      '#2980b9', // Blue
    ];
    
    const negativeColors = [
      '#c0392b', // Dark Red
      '#e74c3c', // Red
      '#d35400', // Burnt Orange
      '#e67e22', // Orange
      '#f39c12', // Gold
    ];
    
    const palette = isPositive ? positiveColors : negativeColors;
    return palette[index % palette.length];
  };
  
  // Map belowThreshold to entries for visual indication
  const enhancedData = data.map((item, index) => ({
    ...item,
    belowThreshold: item.threshold && !item.meetsThreshold,
    // Convert percentage to point value based on total approval points
    pointValue: (item.value / 100) * totalApprovalPoints,
    // Assign color from our accessible palette
    fill: getAccessibleColor(index, type === 'positive')
  }));
  
  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };
  
  const onPieLeave = () => {
    setActiveIndex(null);
  };
  
  return (
    <div className="w-full mb-6">
      <h4 className="text-lg font-semibold mb-2 text-center">
        {title}
      </h4>
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={enhancedData}
              cx="50%"
              cy="50%"
              innerRadius={40}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              startAngle={90}
              endAngle={-270}
              activeIndex={activeIndex}
              activeShape={renderActiveShape}
              onMouseEnter={onPieEnter}
              onMouseLeave={onPieLeave}
            >
              {enhancedData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.fill} 
                  stroke={entry.belowThreshold ? "#c0392b" : "none"}
                  strokeWidth={entry.belowThreshold ? 2 : 0}
                />
              ))}
            </Pie>
            
            {/* Render threshold arrows for entries with thresholds */}
            {enhancedData.filter(entry => entry.threshold && !entry.meetsThreshold).map((entry, index) => {
              const idx = enhancedData.findIndex(e => e.name === entry.name);
              const sumBefore = enhancedData.slice(0, idx).reduce((sum, item) => sum + item.value, 0);
              const middleAngle = 90 - ((sumBefore + entry.value / 2) / 100) * 360;
              
              return (
                <ArrowMarker 
                  key={`threshold-${index}`}
                  cx="50%" 
                  cy="50%"
                  outerRadius={85} 
                  angle={middleAngle}
                  text={`Target: ${entry.threshold.format(entry.threshold.target)}`}
                  isThreshold={true}
                  color="#c0392b"
                />
              );
            })}
            
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                      <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                      <p className="text-gray-600 mb-1">
                        Impact: {Math.abs(data.pointValue).toFixed(1)} points
                        {type === 'positive' ? ' added' : ' reduced'}
                      </p>
                      {data.threshold && (
                        <>
                          <p className="text-gray-600 mb-1">
                            Target: {data.threshold.format(data.threshold.target)}
                          </p>
                          {data.actualValue !== undefined && (
                            <p className={`text-sm ${data.meetsThreshold ? 'text-green-600' : 'text-red-600'}`}>
                              Current: {data.threshold.format(data.actualValue)}
                              <span className="ml-2">
                                {data.meetsThreshold ? '✓' : '✗'}
                              </span>
                            </p>
                          )}
                          <p className="text-xs text-gray-500 mt-2">
                            {data.threshold.description}
                          </p>
                        </>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap justify-center gap-2 mt-2">
        {enhancedData.slice(0, 5).map((entry, index) => (
          <div key={index} className="flex items-center text-xs">
            <div
              className="w-3 h-3 rounded-full mr-1"
              style={{ 
                backgroundColor: entry.fill,
                border: entry.belowThreshold ? '1px solid #c0392b' : 'none'
              }}
            />
            <span className="text-gray-600">
              {entry.name} ({Math.abs(entry.pointValue).toFixed(1)} pts)
              {entry.threshold && (
                <span className={`ml-1 ${entry.meetsThreshold ? 'text-green-600' : 'text-red-600'}`}>
                  {entry.meetsThreshold ? '✓' : '✗'}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const ApprovalChart = ({ featureImportance, baseValue, approvalProbability, approvalThreshold = 0.5 }) => {
  if (!featureImportance || !approvalProbability) return null;

  // Map the threshold-based system to 0-100 scale
  const thresholdPoints = Math.round(approvalThreshold * 100);
  const approvalRatio = approvalProbability / approvalThreshold; // How close to threshold (1.0 = at threshold)
  
  // Calculate normalized points (0-100 scale)
  // If approvalRatio is 1.0 (at threshold), normalizedPoints will be 100
  const normalizedPoints = Math.min(100, Math.round(approvalRatio * 100));
  const isBelowThreshold = approvalProbability < approvalThreshold;
  
  // Get all features with their SHAP values
  const allFeatures = Object.entries(featureImportance);
  
  // Calculate total absolute impact for scaling
  const totalAbsImpact = allFeatures.reduce((sum, [, value]) => sum + Math.abs(value), 0);
  
  // Split into positive and negative features
  const positiveFeatures = allFeatures
    .filter(([, value]) => value > 0)
    .map(([feature, value]) => {
      const baseFeature = getBaseFeature(feature);
      return {
        name: formatFeatureName(feature),
        value: (value / totalAbsImpact) * 100, // Keep percentages for chart segments
        rawValue: value,
        // Scale point value to 0-100 scale
        pointValue: (value / totalAbsImpact) * normalizedPoints,
        fill: COLORS[baseFeature] || COLORS.others,
        featureKey: feature
      };
    })
    .sort((a, b) => b.value - a.value);
  
  const negativeFeatures = allFeatures
    .filter(([, value]) => value < 0)
    .map(([feature, value]) => {
      const baseFeature = getBaseFeature(feature);
      return {
        name: formatFeatureName(feature),
        value: Math.abs(value / totalAbsImpact) * 100, // Keep percentages for chart segments
        rawValue: value,
        // Scale point value to 0-100 scale
        pointValue: Math.abs(value / totalAbsImpact) * normalizedPoints,
        fill: COLORS[baseFeature] || COLORS.others,
        featureKey: feature
      };
    })
    .sort((a, b) => b.value - a.value);

  // Calculate if we need to show threshold gap
  const showThresholdGap = isBelowThreshold;
  
  // Prepare main chart data segments - using 0-100 scale
  const mainChartData = [
    // Current approval (green)
    {
      name: 'Current Approval',
      value: normalizedPoints,
      fill: COLORS.approval,
      isApproval: true,
      noBorder: true
    }
  ];
  
  // Add threshold gap if needed (orange)
  if (showThresholdGap) {
    mainChartData.push({
      name: 'Threshold Gap',
      value: 100 - normalizedPoints,
      fill: COLORS.threshold,
      isThresholdGap: true,
      noBorder: true
    });
  }
  
  // Add target thresholds to chart data
  const addThresholds = (data) => {
    return data.map(item => {
      if (item.featureKey) {
        const baseFeature = getBaseFeature(item.featureKey);
        const threshold = THRESHOLDS[baseFeature];
        
        if (threshold) {
          // Get current value 
          let actualValue = null;
          if (item.featureKey.includes('payment_history_')) {
            actualValue = item.featureKey.includes('On Time') ? 1 : 
                          item.featureKey.includes('Late<30') ? 0.7 : 0.3;
          } else {
            const match = item.featureKey.match(/(\d+(\.\d+)?)/);
            if (match) actualValue = parseFloat(match[1]);
          }
          
          if (actualValue !== null) {
            return {
              ...item,
              threshold,
              actualValue,
              meetsThreshold: threshold.compare(actualValue, threshold.target)
            };
          }
        }
      }
      return item;
    });
  };
  
  // Add thresholds to datasets
  const positiveDataWithThresholds = addThresholds(positiveFeatures);
  const negativeDataWithThresholds = addThresholds(negativeFeatures);

  // Calculate angle for threshold arrow (180 = left side, 0 = right side)
  const thresholdAngle = 0; // Always at the right (100 points)
  
  // Calculate angle for "You are here" arrow (scale based on 0-100)
  const youAreHereAngle = 180 - (normalizedPoints * 180) / 100;

  return (
    <div className="w-full mt-6">
      <h3 className="text-xl font-semibold mb-4 text-center text-gray-800">
        Approval Score Analysis
      </h3>
      
      {/* Main Approval Chart - Using 0-100 scale */}
      <div className="w-full h-[300px] mb-8">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={mainChartData}
              cx="50%"
              cy="50%"
              innerRadius={80}
              outerRadius={120}
              startAngle={180}
              endAngle={0}
              paddingAngle={1}
              dataKey="value"
            >
              {mainChartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.fill} 
                  stroke={entry.hasBorder ? "#c0392b" : "none"}
                  strokeWidth={entry.hasBorder ? 2 : 0}
                />
              ))}
              <Label
                content={<CustomLabel value={normalizedPoints} />}
                position="center"
              />
            </Pie>
            
            {/* Threshold arrow marker - updated text for 100-point scale */}
            <ArrowMarker 
              cx="50%" 
              cy="50%" 
              outerRadius={125} 
              angle={thresholdAngle} 
              text={`Target: 100 pts`}
              isThreshold={true}
              color="#c0392b"
            />
            
            {/* "You are here" marker */}
            <ArrowMarker 
              cx="50%" 
              cy="50%" 
              outerRadius={125} 
              angle={youAreHereAngle} 
              text="You are here"
              isThreshold={false}
              color="#333"
            />
            
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  if (data.isEmpty) return null;
                  
                  if (data.isThresholdGap) {
                    return (
                      <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                        <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                        <p className="text-gray-600">
                          {data.value.toFixed(1)} points needed to reach target
                        </p>
                        <p className="text-xs text-gray-500 mt-2">
                          Points needed to reach the minimum approval threshold
                        </p>
                      </div>
                    );
                  }
                  
                  return (
                    <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                      <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                      <p className="text-gray-600">
                        {data.value.toFixed(1)} points
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex justify-center mt-2 space-x-4">
          <div className="text-center text-sm text-gray-600">
            <span className="inline-block w-3 h-3 bg-[#27ae60] rounded-full mr-1"></span>
            <span>Current: {normalizedPoints} pts</span>
          </div>
          {showThresholdGap && (
            <div className="text-center text-sm text-gray-600">
              <span className="inline-block w-3 h-3 bg-[#d35400] rounded-full mr-1"></span>
              <span>Points Needed: {(100 - normalizedPoints).toFixed(0)}</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        {/* Positive Factors Chart */}
        <FactorPieChart 
          data={positiveDataWithThresholds.slice(0, 5)} 
          title="Factors Adding Points" 
          type="positive"
          approvalThreshold={approvalThreshold}
          totalApprovalPoints={normalizedPoints}
        />
        
        {/* Negative Factors Chart */}
        <FactorPieChart 
          data={negativeDataWithThresholds.slice(0, 5)} 
          title="Factors Reducing Points" 
          type="negative"
          approvalThreshold={approvalThreshold}
          totalApprovalPoints={normalizedPoints}
        />
      </div>
      
      <div className="mt-6 text-center text-gray-600 text-sm max-w-2xl mx-auto p-4 bg-gray-50 rounded-lg">
        <p className="mb-2">
          The main chart shows your current approval score ({normalizedPoints} points).
          {isBelowThreshold ? ` You need ${(100 - normalizedPoints).toFixed(0)} more points to reach the target of 100.` : ' Congratulations! Your score exceeds the minimum target.'}
        </p>
        <p>
          The smaller charts break down how many points each factor is adding or reducing from your score. Factors with red borders need improvement.
        </p>
      </div>
    </div>
  );
};

export default ApprovalChart;