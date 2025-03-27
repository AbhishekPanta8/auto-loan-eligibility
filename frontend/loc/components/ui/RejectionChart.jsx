import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Label } from 'recharts';

const COLORS = {
  'credit_score': '#4ECDC4',
  'DTI': '#45B7D1',
  'payment_history': '#96CEB4',
  'estimated_debt': '#FFEEAD',
  'months_employed': '#FF6B6B',
  'others': '#D4D4D4',
  'empty': '#f5f5f5'  // Light gray for empty space
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

const CustomLabel = ({ viewBox, value }) => {
  const { cx, cy } = viewBox;
  return (
    <g>
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle">
        <tspan x={cx} dy="-1em" fontSize="24" fontWeight="bold">
          {value}%
        </tspan>
        <tspan x={cx} dy="1.5em" fontSize="14" fill="#666">
          Approval Chance
        </tspan>
      </text>
    </g>
  );
};

const ApprovalChart = ({ featureImportance, baseValue, approvalProbability }) => {
  if (!featureImportance || !approvalProbability) return null;

  const approvalPercentage = Math.round(approvalProbability * 100);
  
  // Since the backend is already providing SHAP values reversed for approval
  // Positive values in featureImportance now mean they contribute to approval
  // Negative values in featureImportance now mean they hurt approval
  
  // 1. Calculate total SHAP impact from all features (sum of absolute values)
  const allFeatures = Object.entries(featureImportance);
  const totalAbsImpact = allFeatures.reduce((sum, [, value]) => sum + Math.abs(value), 0);
  
  // 2. Get the top 5 features by absolute impact
  const top5Features = [...allFeatures]
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 5);
    
  // 3. Calculate "Others" as the sum of remaining features
  const otherFeatures = allFeatures
    .filter(([feature]) => !top5Features.some(([topFeature]) => topFeature === feature))
    .reduce((sum, [, value]) => sum + Math.abs(value), 0);
  
  // 4. Calculate the empty portion representing the remaining risk
  const emptyPortion = 100 - approvalPercentage;
  
  // Prepare data for the chart - we'll show the full 100% gauge
  const data = [
    // First add the filled portion (approval chance)
    ...top5Features.map(([feature, value]) => {
      const baseFeature = getBaseFeature(feature);
      const contribution = (Math.abs(value) / totalAbsImpact) * approvalPercentage;
      
      return {
        name: formatFeatureName(feature),
        value: contribution,
        rawShap: value,
        fill: COLORS[baseFeature] || COLORS.others,
        threshold: THRESHOLDS[baseFeature],
        // Since backend has already reversed values, positive now helps approval
        isPositive: value > 0,
        featureKey: feature
      };
    }),
    
    // Add "Others" category if it has a significant contribution
    otherFeatures > 0 ? {
      name: 'Others',
      value: (otherFeatures / totalAbsImpact) * approvalPercentage,
      fill: COLORS.others,
      isOthers: true
    } : null,
    
    // Add empty portion representing rejection chance
    {
      name: 'Remaining Risk',
      value: emptyPortion,
      fill: COLORS.empty,
      isEmpty: true
    }
  ].filter(Boolean); // Remove null items
  
  // Calculate actual values for features from the data for tooltips
  const getActualValue = (feature) => {
    // For payment history features
    if (feature.includes('payment_history_')) {
      if (feature.includes('On Time')) return 1;
      if (feature.includes('Late<30')) return 0.7;
      if (feature.includes('Late>60')) return 0.3;
      return 0.5;
    }
    
    // For normal features, try to extract from the feature name
    const match = feature.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : null;
  };
  
  // Add actual values and threshold checks
  data.forEach(item => {
    if (item.featureKey && item.threshold) {
      const actualValue = getActualValue(item.featureKey);
      if (actualValue !== null) {
        item.actualValue = actualValue;
        item.meetsThreshold = item.threshold.compare(actualValue, item.threshold.target);
      }
    }
  });

  return (
    <div className="w-full h-[400px] mt-6">
      <h3 className="text-xl font-semibold mb-4 text-center text-gray-800">
        Approval Chance Distribution
      </h3>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={80}
            outerRadius={120}
            startAngle={180}
            endAngle={0}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
            <Label
              content={<CustomLabel value={approvalPercentage} />}
              position="center"
            />
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                if (data.isEmpty) return null;
                
                if (data.isOthers) {
                  return (
                    <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                      <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                      <p className="text-gray-600">
                        Contribution: {data.value.toFixed(1)}% of approval chance
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        Combined impact of remaining factors
                      </p>
                    </div>
                  );
                }
                
                return (
                  <div className="bg-white p-4 border rounded shadow-lg max-w-xs">
                    <p className="font-semibold text-gray-800 mb-2">{data.name}</p>
                    <p className="text-gray-600 mb-1">
                      Contribution: {data.value.toFixed(1)}% of approval chance
                      <span className={data.isPositive ? "text-green-600 ml-1" : "text-red-600 ml-1"}>
                        {data.isPositive ? "(Helps)" : "(Hurts)"}
                      </span>
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
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {data.filter(entry => !entry.isEmpty).map((entry, index) => (
          <div key={index} className="flex items-center">
            <div
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: entry.fill }}
            />
            <span className="text-sm text-gray-600">
              {entry.name} ({entry.value.toFixed(1)}%)
              {!entry.isOthers && entry.threshold && (
                <span className={`ml-1 text-xs ${entry.meetsThreshold ? 'text-green-600' : 'text-red-600'}`}>
                  {entry.meetsThreshold ? '✓' : '✗'}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 text-center text-gray-600 text-sm max-w-2xl mx-auto">
        <p className="mb-2">
          The chart shows your overall approval chance ({approvalPercentage}%) and how different factors contribute to it.
        </p>
        <p>
          ✓ indicates meeting target thresholds, while ✗ shows areas needing improvement to increase approval chances.
        </p>
      </div>
    </div>
  );
};

export default ApprovalChart;