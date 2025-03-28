import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Label, Sector } from 'recharts';

const COLORS = {
  'credit_score': '#4ECDC4',
  'DTI': '#45B7D1',
  'payment_history': '#96CEB4',
  'estimated_debt': '#FFEEAD',
  'months_employed': '#FF6B6B',
  'others': '#D4D4D4',
  'empty': '#e0e0e0',        // Light grey for empty space
  'potential': '#e74c3c',    // Changed from yellow (#ffeb3b) to red
  'approval': '#2ecc71',     // Bright green for approval
  'rejection': '#e0e0e0',    // Light grey for rejection
  'threshold': '#e74c3c'     // Changed from yellow (#ffeb3b) to red
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
          {value}/{thresholdValue} pts
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
  
  // Define base colors
  const baseColor = type === 'positive' ? '#2ecc71' : '#e74c3c'; // Green or Red
  
  // Generate color shades based on value proportion
  const getShade = (value, index, total, isPositive) => {
    // For positive values: darker green = higher contribution
    // For negative values: darker red = higher reduction
    const baseHue = isPositive ? 140 : 0; // Green or Red in HSL
    const saturation = 80;
    
    // Get the max value to normalize
    const maxValue = Math.max(...data.map(item => item.value));
    
    // Normalize value to 0-1 range
    const normalizedValue = value / maxValue;
    
    // Higher values get darker colors (lower lightness)
    // Scale from 70% (lightest) to 30% (darkest)
    const lightness = 70 - (normalizedValue * 40);
    
    return `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
  };
  
  // Map belowThreshold to entries for visual indication
  const enhancedData = data.map((item, index) => ({
    ...item,
    belowThreshold: item.threshold && !item.meetsThreshold,
    // Convert percentage to point value based on total approval points
    pointValue: (item.value / 100) * totalApprovalPoints,
    // Assign color shade based on proportional value
    fill: getShade(item.value, index, data.length, type === 'positive')
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
                  stroke={entry.belowThreshold ? "#e74c3c" : "none"}
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
                  color="#e74c3c"
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
                border: entry.belowThreshold ? '1px solid #e74c3c' : 'none'
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

  // Convert percentages to points (removing % symbol)
  const approvalPoints = Math.round(approvalProbability * 100);
  const thresholdPoints = Math.round(approvalThreshold * 100);
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
        value: (value / totalAbsImpact) * 100,
        rawValue: value,
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
        value: Math.abs(value / totalAbsImpact) * 100, // Use absolute value for display
        rawValue: value,
        fill: COLORS[baseFeature] || COLORS.others,
        featureKey: feature
      };
    })
    .sort((a, b) => b.value - a.value);

  // Calculate potential approval chance without negative factors
  const negativeImpact = negativeFeatures.reduce((sum, item) => sum + item.value, 0);
  const potentialApprovalPercentage = Math.min(100, approvalPoints + negativeImpact);
  
  // Calculate if we need to show threshold gap and potential gain segments
  const showThresholdGap = isBelowThreshold && thresholdPoints > approvalPoints;
  const showPotentialGain = potentialApprovalPercentage > thresholdPoints && showThresholdGap;
  
  // Prepare main chart data segments
  const mainChartData = [
    // Current approval (green)
    {
      name: 'Current Approval',
      value: approvalPoints,
      fill: COLORS.approval,
      isApproval: true,
      noBorder: true // No border
    }
  ];
  
  // Add threshold gap if needed (yellow, no border)
  if (showThresholdGap) {
    mainChartData.push({
      name: 'Threshold Gap',
      value: thresholdPoints - approvalPoints,
      fill: COLORS.threshold,
      isThresholdGap: true,
      noBorder: true // No border
    });
  }
  
  // Add potential gain beyond threshold if applicable (yellow with red border)
  if (showPotentialGain) {
    mainChartData.push({
      name: 'Potential Gain',
      value: potentialApprovalPercentage - thresholdPoints,
      fill: COLORS.potential,
      isPotential: true,
      hasBorder: true // Red border
    });
  }
  
  // Add rejection risk (grey)
  mainChartData.push({
    name: 'Rejection Risk',
    value: 100 - Math.max(potentialApprovalPercentage, showThresholdGap ? thresholdPoints : approvalPoints),
    fill: COLORS.rejection,
    isEmpty: true,
    noBorder: true // No border
  });
  
  // Add target thresholds to all chart data
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

  // Calculate angle for threshold arrow
  const thresholdAngle = 180 - (thresholdPoints * 180) / 100;
  
  // Calculate angle for "You are here" arrow
  const youAreHereAngle = 180 - (approvalPoints * 180) / 100;

  return (
    <div className="w-full mt-6">
      <h3 className="text-xl font-semibold mb-4 text-center text-gray-800">
        Approval Score Analysis
      </h3>
      
      {/* Main Approval Chart - Simplified */}
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
                  stroke={entry.hasBorder ? "#e74c3c" : "none"}
                  strokeWidth={entry.hasBorder ? 2 : 0}
                />
              ))}
              <Label
                content={<CustomLabel value={approvalPoints} thresholdValue={thresholdPoints} />}
                position="center"
              />
            </Pie>
            
            {/* Threshold arrow marker - updated text */}
            <ArrowMarker 
              cx="50%" 
              cy="50%" 
              outerRadius={125} 
              angle={thresholdAngle} 
              text={`Target: ${thresholdPoints} pts`}
              isThreshold={true}
              color="#e74c3c"
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
                  
                  if (data.isPotential) {
                    return (
                      <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                        <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                        <p className="text-gray-600">
                          Additional: +{data.value.toFixed(1)} potential points
                        </p>
                        <p className="text-xs text-gray-500 mt-2">
                          Extra points possible by improving all factors
                        </p>
                      </div>
                    );
                  }
                  
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
            <span className="inline-block w-3 h-3 bg-[#2ecc71] rounded-full mr-1"></span>
            <span>Current: {approvalPoints} pts</span>
          </div>
          {showThresholdGap && (
            <div className="text-center text-sm text-gray-600">
              <span className="inline-block w-3 h-3 bg-[#e74c3c] rounded-full mr-1"></span>
              <span>Points Needed: {(thresholdPoints - approvalPoints).toFixed(0)}</span>
            </div>
          )}
          {showPotentialGain && (
            <div className="text-center text-sm text-gray-600">
              <span className="inline-block w-3 h-3 bg-[#e74c3c] rounded-full mr-1" style={{border: '1px solid #e74c3c'}}></span>
              <span>Additional Potential: {(potentialApprovalPercentage - thresholdPoints).toFixed(0)} pts</span>
            </div>
          )}
          <div className="text-center text-sm text-gray-600">
            <span className="inline-block w-3 h-3 bg-[#e0e0e0] rounded-full mr-1"></span>
            <span>Required Points: {(100-Math.max(potentialApprovalPercentage, isBelowThreshold ? thresholdPoints : approvalPoints)).toFixed(0)}</span>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        {/* Positive Factors Chart */}
        <FactorPieChart 
          data={positiveDataWithThresholds.slice(0, 5)} 
          title="Factors Adding Points" 
          type="positive"
          approvalThreshold={approvalThreshold}
          totalApprovalPoints={approvalPoints}
        />
        
        {/* Negative Factors Chart */}
        <FactorPieChart 
          data={negativeDataWithThresholds.slice(0, 5)} 
          title="Factors Reducing Points" 
          type="negative"
          approvalThreshold={approvalThreshold}
          totalApprovalPoints={approvalPoints}
        />
      </div>
      
      <div className="mt-6 text-center text-gray-600 text-sm max-w-2xl mx-auto p-4 bg-gray-50 rounded-lg">
        <p className="mb-2">
          The main chart shows your current approval score ({approvalPoints}/{thresholdPoints} points).
          {isBelowThreshold ? ` You need ${thresholdPoints - approvalPoints} more points to reach the target.` : ' Congratulations! Your score exceeds the target.'}
        </p>
        <p>
          The smaller charts break down how many points each factor is adding or reducing from your score. Factors with red borders need improvement.
        </p>
      </div>
    </div>
  );
};

export default ApprovalChart;