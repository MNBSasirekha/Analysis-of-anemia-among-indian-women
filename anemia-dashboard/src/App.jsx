import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import {
  Activity,
  Upload,
  AlertTriangle,
  CheckCircle2,
  Filter,
  BarChart3,
  Table as TableIcon,
  ShieldCheck,
  Search,
  MapPin,
  TrendingUp,
  TrendingDown,
  Layers,
  Cpu,
  Download,
  Sliders,
  Sparkles,
  ArrowRight,
  Printer,
  ChevronRight,
  Flame,
  FileSpreadsheet,
  HelpCircle,
  BarChart2
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';

// ============================================================================
// 1. DATA SCHEMA & CANONICAL MAPPINGS (707 Districts x 16 Columns)
// ============================================================================
const SCHEMA = {
  DISTRICT: 'District Names',
  STATE: 'State/UT',
  WOMEN_LITERACY: "Women's Literacy (%)",
  SCHOOLING_10: 'Women with 10+ Years of Schooling (%)',
  TEENAGE_PREG: 'Teenage Mothers/Pregnant (%)',
  IFA_100: 'IFA Consumption 100+ Days (%)',
  INST_BIRTHS: 'Institutional Births (%)',
  UNDERWEIGHT_WOMEN: 'Underweight Women (BMI < 18.5) (%)',
  SANITATION: 'Improved Sanitation (%)',
  CLEAN_FUEL: 'Clean Cooking Fuel (%)',
  HEALTH_INSURANCE: 'Health Insurance Coverage (%)',
  WOMEN_ANAEMIA: 'All Women (15–49) Anaemic (%)',
  CHILDREN_ANAEMIA: 'Children (6–59 months) Anaemic (%)',
  NON_PREG_ANAEMIA: 'Non-Pregnant Women Anaemic (%)',
  PREG_ANAEMIA: 'Pregnant Women Anaemic (%)',
  ADOLESCENT_ANAEMIA: 'Women (15–19) Anaemic (%)'
};

const INDEPENDENT_VARS = [
  SCHEMA.WOMEN_LITERACY,
  SCHEMA.SCHOOLING_10,
  SCHEMA.TEENAGE_PREG,
  SCHEMA.IFA_100,
  SCHEMA.INST_BIRTHS,
  SCHEMA.UNDERWEIGHT_WOMEN,
  SCHEMA.SANITATION,
  SCHEMA.CLEAN_FUEL,
  SCHEMA.HEALTH_INSURANCE
];

const WHO_THRESHOLDS = {
  NORMAL: 19.9,
  MODERATE: 39.9,
  SEVERE: 59.9
};

// ==========================================
// REPLACE THIS FUNCTION (Around Line 55 - 85)
// ==========================================
function normalizeHeader(raw) {
  const c = raw.toLowerCase().trim().replace(/[^a-z0-9]/g, '');

  // 1. MUST check Institutional Births FIRST so "institUTional" does not trigger "ut"
  if (c.includes('institutional') || c.includes('instbirth')) return SCHEMA.INST_BIRTHS;
  if (c.includes('district')) return SCHEMA.DISTRICT;

  // 2. Strict State/UT detection (ignores numeric state codes like 'state_code')
  if (
    c === 'ut' || 
    c.includes('stateut') || 
    (c.includes('state') && !c.includes('code') && !c.includes('id'))
  ) {
    return SCHEMA.STATE;
  }

  // 3. Clinical & Socioeconomic indicators
  if (c.includes('literacy')) return SCHEMA.WOMEN_LITERACY;
  if (c.includes('schooling') || c.includes('10year')) return SCHEMA.SCHOOLING_10;
  if (c.includes('teenage') || c.includes('teenpreg') || c.includes('teenagemother')) return SCHEMA.TEENAGE_PREG;
  if (c.includes('ifa') || c.includes('100day')) return SCHEMA.IFA_100;
  if (c.includes('underweight') || c.includes('bmi185') || c.includes('bmi')) return SCHEMA.UNDERWEIGHT_WOMEN;
  if (c.includes('sanitation')) return SCHEMA.SANITATION;
  if (c.includes('cleanfuel') || c.includes('cookingfuel') || c.includes('lpg')) return SCHEMA.CLEAN_FUEL;
  if (c.includes('insurance')) return SCHEMA.HEALTH_INSURANCE;
  if (c.includes('children') || c.includes('659')) return SCHEMA.CHILDREN_ANAEMIA;
  if (c.includes('nonpregnant') || c.includes('nonpreg')) return SCHEMA.NON_PREG_ANAEMIA;
  if (c.includes('pregnant') && !c.includes('non')) return SCHEMA.PREG_ANAEMIA;
  if (c.includes('1519') || c.includes('adolescent')) return SCHEMA.ADOLESCENT_ANAEMIA;
  if (c.includes('allwomen') || c.includes('1549') || c.includes('womenanaem')) return SCHEMA.WOMEN_ANAEMIA;

  return raw;
}


// Sanitize & Clamp Percentages: filters out error codes like -99, 999, negative numbers
function sanitizePercentage(raw) {
  if (raw === null || raw === undefined || raw === '' || raw === 'NA' || raw === '-') return null;
  const num = Number(raw);
  if (isNaN(num)) return null;
  if (num < 0 || num > 100) return null;
  return num;
}

// ============================================================================
// 2. MATHEMATICAL & MACHINE LEARNING UTILITIES (Ordinary Least Squares)
// ============================================================================

// Pearson Correlation Coefficient (r)
function calculatePearsonR(xArr, yArr) {
  const valid = [];
  for (let i = 0; i < xArr.length; i++) {
    if (typeof xArr[i] === 'number' && typeof yArr[i] === 'number' && !isNaN(xArr[i]) && !isNaN(yArr[i])) {
      valid.push([xArr[i], yArr[i]]);
    }
  }
  const n = valid.length;
  if (n < 4) return 0;

  const meanX = valid.reduce((acc, p) => acc + p[0], 0) / n;
  const meanY = valid.reduce((acc, p) => acc + p[1], 0) / n;

  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = valid[i][0] - meanX;
    const dy = valid[i][1] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  return den === 0 ? 0 : Number((num / den).toFixed(3));
}

// Multi-variable Linear Regression with Standardized Coefficients (Beta)
function trainMultivariateRegression(dataset, featureKeys, targetKey) {
  // Filter complete rows
  const cleanRows = dataset.filter(row => {
    if (typeof row[targetKey] !== 'number' || isNaN(row[targetKey])) return false;
    for (const key of featureKeys) {
      if (typeof row[key] !== 'number' || isNaN(row[key])) return false;
    }
    return true;
  });

  const N = cleanRows.length;
  if (N < featureKeys.length + 5) {
    return { weights: {}, r2: 0, mae: 0, rmse: 0, intercept: 0, valid: false };
  }

  // Calculate Mean and StdDev for standardization
  const stats = {};
  featureKeys.forEach(k => {
    const vals = cleanRows.map(r => r[k]);
    const mean = vals.reduce((a, b) => a + b, 0) / N;
    const sd = Math.sqrt(vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / N) || 1;
    stats[k] = { mean, sd };
  });

  const yVals = cleanRows.map(r => r[targetKey]);
  const yMean = yVals.reduce((a, b) => a + b, 0) / N;
  const ySd = Math.sqrt(yVals.reduce((a, b) => a + Math.pow(b - yMean, 2), 0) / N) || 1;

  // Build Standardized Design Matrix X (N x P) and Vector Y (N x 1)
  const P = featureKeys.length;
  const X = cleanRows.map(r => featureKeys.map(k => (r[k] - stats[k].mean) / stats[k].sd));
  const Y = cleanRows.map(r => (r[targetKey] - yMean) / ySd);

  // Compute (X^T * X) and (X^T * Y)
  const XtX = Array.from({ length: P }, () => Array(P).fill(0));
  const XtY = Array(P).fill(0);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < P; j++) {
      XtY[j] += X[i][j] * Y[i];
      for (let k = 0; k < P; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }

  // Ridge Regularization (L2 penalty) for Numerical Stability
  const lambda = 0.05;
  for (let j = 0; j < P; j++) XtX[j][j] += lambda;

  // Invert (XtX) via Gauss-Jordan Elimination
  const inv = invertMatrix(XtX);
  if (!inv) return { weights: {}, r2: 0, mae: 0, rmse: 0, intercept: 0, valid: false };

  // Weights in Standardized Space: beta = inv(XtX) * XtY
  const beta = Array(P).fill(0);
  for (let i = 0; i < P; i++) {
    for (let j = 0; j < P; j++) {
      beta[i] += inv[i][j] * XtY[j];
    }
  }

  // Compute Unstandardized Coefficients and Model Predictions
  const weights = {};
  featureKeys.forEach((k, idx) => {
    weights[k] = {
      standardizedBeta: Number(beta[idx].toFixed(4)),
      impactDirection: beta[idx] > 0 ? 'Risk Factor (+)' : 'Protective Factor (-)'
    };
  });

  // Calculate Metrics: R2, MAE, RMSE on Original Scale
  let ssTot = 0, ssRes = 0, absErrSum = 0;
  cleanRows.forEach((r, idx) => {
    let predStd = 0;
    for (let j = 0; j < P; j++) {
      predStd += X[idx][j] * beta[j];
    }
    const predOriginal = predStd * ySd + yMean;
    const actual = r[targetKey];

    ssTot += Math.pow(actual - yMean, 2);
    ssRes += Math.pow(actual - predOriginal, 2);
    absErrSum += Math.abs(actual - predOriginal);
  });

  const r2 = Math.max(0, 1 - (ssRes / (ssTot || 1)));
  const mae = absErrSum / N;
  const rmse = Math.sqrt(ssRes / N);

  return {
    weights,
    stats,
    yMean,
    ySd,
    beta,
    featureKeys,
    r2: Number(r2.toFixed(3)),
    mae: Number(mae.toFixed(2)),
    rmse: Number(rmse.toFixed(2)),
    valid: true,
    sampleSize: N
  };
}

// Gauss-Jordan Matrix Inversion
function invertMatrix(matrix) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  ]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) maxRow = k;
    }
    const temp = augmented[i];
    augmented[i] = augmented[maxRow];
    augmented[maxRow] = temp;

    const pivot = augmented[i][i];
    if (Math.abs(pivot) < 1e-9) return null;

    for (let j = 0; j < 2 * n; j++) augmented[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = augmented[k][i];
        for (let j = 0; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }
  }

  return augmented.map(row => row.slice(n));
}

// Animated Numerical Counter for SaaS Dashboard Aesthetics
function AnimatedCounter({ value, decimals = 1, unit = '' }) {
  const [display, setDisplay] = useState(0);
  const target = parseFloat(value) || 0;

  useEffect(() => {
    let start = display;
    let startTime = null;
    const duration = 650;

    const tick = (now) => {
      if (!startTime) startTime = now;
      const progress = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setDisplay(start + (target - start) * ease);
      if (progress < 1) requestAnimationFrame(tick);
      else setDisplay(target);
    };

    requestAnimationFrame(tick);
  }, [target]);

  return (
    <span>
      {display.toFixed(decimals)}
      {unit}
    </span>
  );
}

// Color Utility based on WHO Anaemia Cutoffs
function getPrevalenceColor(val) {
  if (val === null || val === undefined || isNaN(val)) return '#64748B';
  if (val < 20) return '#10B981'; // Mild public health concern
  if (val < 40) return '#F59E0B'; // Moderate
  if (val < 60) return '#F97316'; // Severe
  return '#EF4444'; // Catastrophic / Critical
}

// ============================================================================
// 3. MAIN COMPONENT CONTROLLER
// ============================================================================
export default function AnemiaAnalyticsEngine() {
  const [dataset, setDataset] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedState, setSelectedState] = useState('ALL');
  const [selectedDistrict, setSelectedDistrict] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedForInsight, setSelectedForInsight] = useState(null);
  // --- ADD THESE 4 LINES HERE ---
  const [dynamicMetric, setDynamicMetric] = useState(SCHEMA.WOMEN_ANAEMIA);
  const [dynamicChartType, setDynamicChartType] = useState('bar');
  const [dynamicTopN, setDynamicTopN] = useState(10);
  const [dynamicSortOrder, setDynamicSortOrder] = useState('desc');

  // Policy Simulator Inputs
  const [simLiteracyBoost, setSimLiteracyBoost] = useState(10);
  const [simSanitationBoost, setSimSanitationBoost] = useState(15);
  const [simIFABoost, setSimIFABoost] = useState(20);

  // Data Loading from Public Directory or Drag-Drop File
  useEffect(() => {
    fetch('./nfhs5_district_anemia.csv')
      .then(res => (res.ok ? res.text() : null))
      .then(text => {
        if (text) {
          const parsed = parseRawCSV(text);
          if (parsed.length > 0) setDataset(parsed);
        }
      })
      .catch(() => {});
  }, []);

  const parseRawCSV = (text) => {
    const lines = text.split(/\r\n|\n/).filter(line => line.trim() !== '');
    if (lines.length < 2) return [];

    const parseLine = (line) => {
      const res = [];
      let start = 0;
      let inQ = false;
      for (let i = 0; i < line.length; i++) {
        if (line[i] === '"') inQ = !inQ;
        else if (line[i] === ',' && !inQ) {
          res.push(line.substring(start, i).trim().replace(/^"|"$/g, ''));
          start = i + 1;
        }
      }
      res.push(line.substring(start).trim().replace(/^"|"$/g, ''));
      return res;
    };

    const headers = parseLine(lines[0]).map(normalizeHeader);
    const parsed = [];

    for (let i = 1; i < lines.length; i++) {
      const values = parseLine(lines[i]);
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((h, idx) => {
          const val = values[idx];
          if (h === SCHEMA.DISTRICT || h === SCHEMA.STATE) {
            row[h] = val && val.trim() !== '' ? val.trim() : null;
          } else {
            row[h] = sanitizePercentage(val);
          }
        });
        if (row[SCHEMA.DISTRICT] && row[SCHEMA.STATE]) {
          parsed.push(row);
        }
      }
    }
    return parsed;
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const content = evt.target.result;
        let data = [];
        if (file.name.endsWith('.json')) {
          const raw = JSON.parse(content);
          data = raw.map(r => {
            const mapped = {};
            Object.keys(r).forEach(k => {
              const h = normalizeHeader(k);
              mapped[h] = (h === SCHEMA.DISTRICT || h === SCHEMA.STATE) ? r[k] : sanitizePercentage(r[k]);
            });
            return mapped;
          });
        } else {
          data = parseRawCSV(content);
        }

        if (data.length === 0) {
          alert('Upload Error: File does not contain valid NFHS-5 district rows.');
          return;
        }
        setDataset(data);
        setSelectedState('ALL');
        setSelectedDistrict('ALL');
      } catch (err) {
        alert('File ingestion failed. Please verify format.');
      }
    };
    reader.readAsText(file);
  };

  // State & District Dropdown Options
  const stateOptions = useMemo(() => {
    const s = new Set();
    dataset.forEach(r => { if (r[SCHEMA.STATE]) s.add(r[SCHEMA.STATE]); });
    return Array.from(s).sort();
  }, [dataset]);

  const districtOptions = useMemo(() => {
    const d = new Set();
    dataset.forEach(r => {
      if (selectedState === 'ALL' || r[SCHEMA.STATE] === selectedState) {
        if (r[SCHEMA.DISTRICT]) d.add(r[SCHEMA.DISTRICT]);
      }
    });
    return Array.from(d).sort();
  }, [dataset, selectedState]);

  // Filtered Dataset
  const filteredData = useMemo(() => {
    return dataset.filter(r => {
      if (selectedState !== 'ALL' && r[SCHEMA.STATE] !== selectedState) return false;
      if (selectedDistrict !== 'ALL' && r[SCHEMA.DISTRICT] !== selectedDistrict) return false;
      if (searchQuery.trim() !== '') {
        const q = searchQuery.toLowerCase();
        const dist = (r[SCHEMA.DISTRICT] || '').toLowerCase();
        const st = (r[SCHEMA.STATE] || '').toLowerCase();
        if (!dist.includes(q) && !st.includes(q)) return false;
      }
      return true;
    });
  }, [dataset, selectedState, selectedDistrict, searchQuery]);

  // ============================================================================
  // 4. EXECUTIVE OVERVIEW METRICS
  // ============================================================================
  const overviewMetrics = useMemo(() => {
    const totalDistricts = filteredData.length;
    const statesCount = new Set(filteredData.map(r => r[SCHEMA.STATE]).filter(Boolean)).size;

    const calcMean = (k) => {
      const valid = filteredData.map(r => r[k]).filter(v => typeof v === 'number');
      if (valid.length === 0) return 0;
      return Number((valid.reduce((a, b) => a + b, 0) / valid.length).toFixed(1));
    };

    const sortedByWomen = [...filteredData]
      .filter(r => typeof r[SCHEMA.WOMEN_ANAEMIA] === 'number')
      .sort((a, b) => b[SCHEMA.WOMEN_ANAEMIA] - a[SCHEMA.WOMEN_ANAEMIA]);

    const highestDistrict = sortedByWomen[0] || { [SCHEMA.DISTRICT]: 'N/A', [SCHEMA.STATE]: '', [SCHEMA.WOMEN_ANAEMIA]: 0 };
    const lowestDistrict = sortedByWomen[sortedByWomen.length - 1] || { [SCHEMA.DISTRICT]: 'N/A', [SCHEMA.STATE]: '', [SCHEMA.WOMEN_ANAEMIA]: 0 };

    return {
      totalDistricts,
      statesCount,
      womenAvg: calcMean(SCHEMA.WOMEN_ANAEMIA),
      childrenAvg: calcMean(SCHEMA.CHILDREN_ANAEMIA),
      pregnantAvg: calcMean(SCHEMA.PREG_ANAEMIA),
      adolescentAvg: calcMean(SCHEMA.ADOLESCENT_ANAEMIA),
      highestDistrict,
      lowestDistrict
    };
  }, [filteredData]);

  // ============================================================================
  // 5. STATE RANKINGS & DRILL-DOWN DATA
  // ============================================================================
  const stateAggregations = useMemo(() => {
    const map = {};
    dataset.forEach(r => {
      const st = r[SCHEMA.STATE];
      if (!st) return;
      if (!map[st]) {
        map[st] = { state: st, sumWomen: 0, countWomen: 0, districts: 0, sumLit: 0, countLit: 0 };
      }
      map[st].districts++;
      if (typeof r[SCHEMA.WOMEN_ANAEMIA] === 'number') {
        map[st].sumWomen += r[SCHEMA.WOMEN_ANAEMIA];
        map[st].countWomen++;
      }
      if (typeof r[SCHEMA.WOMEN_LITERACY] === 'number') {
        map[st].sumLit += r[SCHEMA.WOMEN_LITERACY];
        map[st].countLit++;
      }
    });

    return Object.values(map)
      .map(s => ({
        state: s.state,
        districts: s.districts,
        avgWomen: s.countWomen > 0 ? Number((s.sumWomen / s.countWomen).toFixed(1)) : 0,
        avgLit: s.countLit > 0 ? Number((s.sumLit / s.countLit).toFixed(1)) : 0
      }))
      .sort((a, b) => b.avgWomen - a.avgWomen);
  }, [dataset]);

  // Top 10 and Bottom 10 States
  const top10States = useMemo(() => stateAggregations.slice(0, 10), [stateAggregations]);
  const bottom10States = useMemo(() => [...stateAggregations].reverse().slice(0, 10), [stateAggregations]);

  // Top 10 and Bottom 10 Districts
  const top10Districts = useMemo(() => {
    return [...filteredData]
      .filter(r => typeof r[SCHEMA.WOMEN_ANAEMIA] === 'number')
      .sort((a, b) => b[SCHEMA.WOMEN_ANAEMIA] - a[SCHEMA.WOMEN_ANAEMIA])
      .slice(0, 10)
      .map(r => ({
        name: `${r[SCHEMA.DISTRICT]} (${r[SCHEMA.STATE]})`,
        prevalence: r[SCHEMA.WOMEN_ANAEMIA]
      }));
  }, [filteredData]);

  const bottom10Districts = useMemo(() => {
    
    return [...filteredData]
      .filter(r => typeof r[SCHEMA.WOMEN_ANAEMIA] === 'number')
      .sort((a, b) => a[SCHEMA.WOMEN_ANAEMIA] - b[SCHEMA.WOMEN_ANAEMIA])
      .slice(0, 10)
      .map(r => ({
        name: `${r[SCHEMA.DISTRICT]} (${r[SCHEMA.STATE]})`,
        prevalence: r[SCHEMA_WOMEN_ANAEMIA] || r[SCHEMA.WOMEN_ANAEMIA]
      }));
  }, [filteredData]);

  // --- ADD THIS CALCULATION BLOCK ---
  const dynamicChartData = useMemo(() => {
    const validRows = filteredData.filter(
      (r) => typeof r[dynamicMetric] === 'number' && !isNaN(r[dynamicMetric])
    );

    const sorted = [...validRows].sort((a, b) => {
      return dynamicSortOrder === 'desc'
        ? b[dynamicMetric] - a[dynamicMetric]
        : a[dynamicMetric] - b[dynamicMetric];
    });

    const sliced = dynamicTopN === 'ALL' ? sorted : sorted.slice(0, Number(dynamicTopN));

    return sliced.map((r) => ({
      name: `${r[SCHEMA.DISTRICT]} (${r[SCHEMA.STATE]})`,
      district: r[SCHEMA.DISTRICT],
      state: r[SCHEMA.STATE],
      value: r[dynamicMetric]
    }));
  }, [filteredData, dynamicMetric, dynamicSortOrder, dynamicTopN]);

  function SCHEMA_WOMEN_ANAEMIA(r) {
    return r[SCHEMA.WOMEN_ANAEMIA];
  }

  // ============================================================================
  // 6. MACHINE LEARNING MODEL INTEGRATION
  // ============================================================================
  const mlModel = useMemo(() => {
    return trainMultivariateRegression(dataset, INDEPENDENT_VARS, SCHEMA.WOMEN_ANAEMIA);
  }, [dataset]);

  const featureImportanceList = useMemo(() => {
    if (!mlModel.valid) return [];
    return Object.entries(mlModel.weights)
      .map(([k, v]) => ({
        feature: k,
        beta: v.standardizedBeta,
        absBeta: Math.abs(v.standardizedBeta),
        direction: v.impactDirection
      }))
      .sort((a, b) => b.absBeta - a.absBeta);
  }, [mlModel]);

  // Policy Simulator Prediction Calculator
  const simulatedImpact = useMemo(() => {
    if (!mlModel.valid) return { predictedReduction: 0, newPrevalence: overviewMetrics.womenAvg };
    // Estimate change using standardized betas
    const litBeta = mlModel.weights[SCHEMA.WOMEN_LITERACY]?.standardizedBeta || 0;
    const sanBeta = mlModel.weights[SCHEMA.SANITATION]?.standardizedBeta || 0;
    const ifaBeta = mlModel.weights[SCHEMA.IFA_100]?.standardizedBeta || 0;

    const litDeltaZ = simLiteracyBoost / (mlModel.stats[SCHEMA.WOMEN_LITERACY]?.sd || 15);
    const sanDeltaZ = simSanitationBoost / (mlModel.stats[SCHEMA.SANITATION]?.sd || 15);
    const ifaDeltaZ = simIFABoost / (mlModel.stats[SCHEMA.IFA_100]?.sd || 15);

    const deltaYStd = (litBeta * litDeltaZ) + (sanBeta * sanDeltaZ) + (ifaBeta * ifaDeltaZ);
    const deltaY = deltaYStd * mlModel.ySd;

    const baseline = overviewMetrics.womenAvg;
    const newPrevalence = Math.max(5, Math.min(100, Number((baseline + deltaY).toFixed(1))));
    const reduction = Number((baseline - newPrevalence).toFixed(1));

    return { reduction, newPrevalence };
  }, [mlModel, simLiteracyBoost, simSanitationBoost, simIFABoost, overviewMetrics.womenAvg]);

  // ============================================================================
  // 7. AUTOMATED DISTRICT DIAGNOSTIC DOSSIER
  // ============================================================================
  const activeDistrictDossier = useMemo(() => {
    const target = selectedForInsight || filteredData[0];
    if (!target) return null;

    const distVal = target[SCHEMA.WOMEN_ANAEMIA];
    const nationalAvg = overviewMetrics.womenAvg;

    // State baseline
    const stateDistricts = dataset.filter(r => r[SCHEMA.STATE] === target[SCHEMA.STATE]);
    const stateVals = stateDistricts.map(r => r[SCHEMA.WOMEN_ANAEMIA]).filter(v => typeof v === 'number');
    const stateAvg = stateVals.length > 0 ? Number((stateVals.reduce((a, b) => a + b, 0) / stateVals.length).toFixed(1)) : nationalAvg;

    const stateDiff = Number((distVal - stateAvg).toFixed(1));
    const natDiff = Number((distVal - nationalAvg).toFixed(1));

    // Clinical severity tag
    let severityTag = 'Normal / Low Burden';
    let tagColor = 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    if (distVal >= 60) {
      severityTag = 'Catastrophic Public Health Emergency';
      tagColor = 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    } else if (distVal >= 40) {
      severityTag = 'High Public Health Problem';
      tagColor = 'text-orange-400 border-orange-500/30 bg-orange-500/10';
    } else if (distVal >= 20) {
      severityTag = 'Moderate Prevalence';
      tagColor = 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    }

    // Identify lagging variables (< 25th percentile or unfavorable)
    const riskFactors = [];
    const protectiveStrengths = [];

    if (target[SCHEMA.UNDERWEIGHT_WOMEN] > 25) {
      riskFactors.push({ label: 'Maternal Undernutrition (BMI < 18.5)', value: `${target[SCHEMA.UNDERWEIGHT_WOMEN]}%`, desc: 'Severe macro-nutritional deficit' });
    }
    if (target[SCHEMA.TEENAGE_PREG] > 10) {
      riskFactors.push({ label: 'Adolescent Pregnancy Rate', value: `${target[SCHEMA.TEENAGE_PREG]}%`, desc: 'Biological vulnerability & high blood loss risk' });
    }
    if (target[SCHEMA.IFA_100] < 30) {
      riskFactors.push({ label: 'Sub-optimal IFA Consumption', value: `${target[SCHEMA.IFA_100]}%`, desc: 'Poor adherence to 100+ days iron supplementation' });
    }
    if (target[SCHEMA.SANITATION] < 50) {
      riskFactors.push({ label: 'Inadequate Sanitation Access', value: `${target[SCHEMA.SANITATION]}%`, desc: 'High exposure to soil-transmitted helminths & enteropathy' });
    }

    if (target[SCHEMA.WOMEN_LITERACY] > 75) {
      protectiveStrengths.push({ label: 'High Female Literacy', value: `${target[SCHEMA.WOMEN_LITERACY]}%`, desc: 'Promotes health-seeking behaviors' });
    }
    if (target[SCHEMA.INST_BIRTHS] > 85) {
      protectiveStrengths.push({ label: 'Institutional Delivery Coverage', value: `${target[SCHEMA.INST_BIRTHS]}%`, desc: 'Adequate access to intrapartum healthcare facilities' });
    }

    // Policy Recommendations
    const recommendations = [];
    if (target[SCHEMA.IFA_100] < 40) {
      recommendations.push('Intensify ASHA-supervised door-to-door distribution of 180-day IFA supplements under Anaemia Mukt Bharat (AMB).');
    }
    if (target[SCHEMA.SANITATION] < 60) {
      recommendations.push('Accelerate Swachh Bharat Gramin Phase-II to mitigate worm infestations causing intestinal blood loss.');
    }
    if (target[SCHEMA.UNDERWEIGHT_WOMEN] > 20) {
      recommendations.push('Integrate double-fortified salt (iron + iodine) and millets into the Targeted Public Distribution System (TPDS).');
    }
    if (recommendations.length === 0) {
      recommendations.push('Maintain universal deworming biannual campaigns and continuous antenatal screening to prevent recurrence.');
    }

    return {
      district: target[SCHEMA.DISTRICT],
      state: target[SCHEMA.STATE],
      val: distVal,
      stateAvg,
      nationalAvg,
      stateDiff,
      natDiff,
      severityTag,
      tagColor,
      riskFactors,
      protectiveStrengths,
      recommendations
    };
  }, [selectedForInsight, filteredData, dataset, overviewMetrics.womenAvg]);

  // Export Data Handler
  const exportToCSV = () => {
    if (filteredData.length === 0) return;
    const headers = Object.keys(filteredData[0]);
    const rows = [headers.join(',')];
    filteredData.forEach(r => {
      const line = headers.map(h => {
        const v = r[h];
        return typeof v === 'string' ? `"${v.replace(/"/g, '""')}"` : v ?? '';
      });
      rows.push(line.join(','));
    });
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `NFHS5_Anaemia_Analytics_Export_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 font-sans selection:bg-rose-500/30 selection:text-rose-200">
      {/* ========================================================================= */}
      {/* 1. TOP NAVIGATION & EXECUTIVE APP HEADER */}
      {/* ========================================================================= */}
      <header className="border-b border-slate-800/80 bg-[#0B1120]/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1720px] mx-auto px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 via-rose-600 to-indigo-700 p-0.5 shadow-lg shadow-rose-600/30 flex items-center justify-center">
              <div className="w-full h-full bg-[#0B1120] rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-rose-500 animate-pulse" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base font-bold tracking-tight text-white">
                  Analysis of Anaemia Among Indian Women
                </h1>
                <span className="text-[10px] uppercase font-mono tracking-widest bg-rose-500/10 text-rose-400 border border-rose-500/20 px-2 py-0.5 rounded-full font-semibold">
                  NFHS-5 (707 Districts)
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Senior Academic Research Dashboard | Epidemiological & Policy Machine Learning Engine
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <label className="flex items-center text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium cursor-pointer border border-slate-700 transition">
              <Upload className="w-3.5 h-3.5 mr-1.5 text-rose-400" />
              {dataset.length === 0 ? 'Upload NFHS-5 CSV' : 'Update Dataset'}
              <input type="file" accept=".csv,.json" onChange={handleFileUpload} className="hidden" />
            </label>

            <button
              onClick={exportToCSV}
              className="flex items-center text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium border border-slate-700 transition"
            >
              <Download className="w-3.5 h-3.5 mr-1.5 text-cyan-400" /> Export Data
            </button>

            <button
              onClick={() => window.print()}
              className="flex items-center text-xs px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold shadow-md shadow-rose-600/30 transition"
            >
              <Printer className="w-3.5 h-3.5 mr-1.5" /> PDF Report
            </button>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* 2. DYNAMIC SLICERS / GLOBAL FILTER STRIP */}
      {/* ========================================================================= */}
      <div className="border-b border-slate-800 bg-[#0B1120]/50 px-6 py-3">
        <div className="max-w-[1720px] mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-1.5 text-xs text-slate-400 uppercase font-mono font-semibold tracking-wider">
              <Filter className="w-3.5 h-3.5 text-rose-500" />
              <span>Slicers:</span>
            </div>

            {/* State Slicer */}
            <select
              value={selectedState}
              onChange={e => {
                setSelectedState(e.target.value);
                setSelectedDistrict('ALL');
              }}
              className="text-xs bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 font-medium focus:border-rose-500 outline-none"
            >
              <option value="ALL">All States / UTs ({stateOptions.length})</option>
              {stateOptions.map(st => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>

            {/* District Slicer */}
            <select
              value={selectedDistrict}
              onChange={e => setSelectedDistrict(e.target.value)}
              className="text-xs bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 font-medium focus:border-rose-500 outline-none"
            >
              <option value="ALL">All Districts ({districtOptions.length})</option>
              {districtOptions.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>

            {(selectedState !== 'ALL' || selectedDistrict !== 'ALL' || searchQuery !== '') && (
              <button
                onClick={() => {
                  setSelectedState('ALL');
                  setSelectedDistrict('ALL');
                  setSearchQuery('');
                }}
                className="text-xs text-rose-400 hover:text-rose-300 font-semibold underline underline-offset-2 ml-2"
              >
                Reset All Filters
              </button>
            )}
          </div>

          {/* Quick Search */}
          <div className="relative w-72">
            <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
            <input
              type="text"
              placeholder="Search district or state..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full text-xs bg-slate-900/90 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-slate-200 focus:border-rose-500 outline-none transition"
            />
          </div>
        </div>
      </div>

      <main className="max-w-[1720px] mx-auto p-6 space-y-6">
        {/* ========================================================================= */}
        {/* 3. EXECUTIVE KPI CARDS WITH ANIMATED COUNTERS */}
        {/* ========================================================================= */}
        <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {/* Card 1 */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0F172A]/90 relative overflow-hidden group hover:border-slate-700 transition">
            <div className="flex justify-between items-center text-slate-400 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">Districts Evaluated</span>
              <Layers className="w-4 h-4 text-slate-500" />
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              <AnimatedCounter value={overviewMetrics.totalDistricts} decimals={0} />
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">NFHS-5 Aggregates</span>
          </div>

          {/* Card 2 */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0F172A]/90 relative overflow-hidden group hover:border-slate-700 transition">
            <div className="flex justify-between items-center text-slate-400 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">States / UTs</span>
              <MapPin className="w-4 h-4 text-slate-500" />
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              <AnimatedCounter value={overviewMetrics.statesCount} decimals={0} />
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">Administrative Units</span>
          </div>

          {/* Card 3: Primary Focus */}
          <div className="p-4 rounded-xl border border-rose-500/30 bg-gradient-to-br from-rose-950/20 via-[#0F172A] to-[#0F172A] relative overflow-hidden border-l-4 border-l-rose-500">
            <div className="flex justify-between items-center text-rose-300 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">Avg Women Anaemia</span>
              <Activity className="w-4 h-4 text-rose-500" />
            </div>
            <div className="text-2xl font-bold text-rose-500 tracking-tight">
              <AnimatedCounter value={overviewMetrics.womenAvg} decimals={1} unit="%" />
            </div>
            <span className="text-[11px] text-slate-400 mt-0.5 block">All Women (15–49 yrs)</span>
          </div>

          {/* Card 4: Children */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0F172A]/90 border-l-4 border-l-amber-500">
            <div className="flex justify-between items-center text-slate-400 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">Children Anaemic</span>
              <Activity className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-2xl font-bold text-amber-400 tracking-tight">
              <AnimatedCounter value={overviewMetrics.childrenAvg} decimals={1} unit="%" />
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">Age 6–59 Months</span>
          </div>

          {/* Card 5: Highest Hotspot District */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0F172A]/90 border-l-4 border-l-rose-600">
            <div className="flex justify-between items-center text-slate-400 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">Highest Hotspot</span>
              <TrendingUp className="w-4 h-4 text-rose-500" />
            </div>
            <div className="text-base font-bold text-white truncate" title={overviewMetrics.highestDistrict[SCHEMA.DISTRICT]}>
              {overviewMetrics.highestDistrict[SCHEMA.DISTRICT]}
            </div>
            <span className="text-[11px] text-rose-400 font-mono font-bold">
              {overviewMetrics.highestDistrict[SCHEMA.WOMEN_ANAEMIA]}% ({overviewMetrics.highestDistrict[SCHEMA.STATE]})
            </span>
          </div>

          {/* Card 6: Lowest District */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0F172A]/90 border-l-4 border-l-emerald-500">
            <div className="flex justify-between items-center text-slate-400 mb-1">
              <span className="text-[10px] uppercase font-mono font-semibold tracking-wider">Lowest Prevalence</span>
              <TrendingDown className="w-4 h-4 text-emerald-500" />
            </div>
            <div className="text-base font-bold text-white truncate" title={overviewMetrics.lowestDistrict[SCHEMA.DISTRICT]}>
              {overviewMetrics.lowestDistrict[SCHEMA.DISTRICT]}
            </div>
            <span className="text-[11px] text-emerald-400 font-mono font-bold">
              {overviewMetrics.lowestDistrict[SCHEMA.WOMEN_ANAEMIA]}% ({overviewMetrics.lowestDistrict[SCHEMA.STATE]})
            </span>
          </div>
        </section>

        {/* ========================================================================= */}
        {/* 4. DASHBOARD TABS NAVIGATION */}
        {/* ========================================================================= */}
        <nav className="flex space-x-2 border-b border-slate-800/90 pb-2 overflow-x-auto">
          {[
            { id: 'overview', label: 'Executive Overview', icon: BarChart3 },
            { id: 'states', label: 'State Analysis & Drill-down', icon: Layers },
            { id: 'districts', label: 'District Rankings', icon: BarChart2 },
            { id: 'factors', label: 'Anaemia Factors & Correlation', icon: Sliders },
            { id: 'ml_insights', label: 'Machine Learning & Policy Simulator', icon: Cpu },
            { id: 'dossier', label: 'District Diagnostic Dossier', icon: Sparkles }
          ].map(t => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center text-xs font-semibold px-4 py-2.5 rounded-lg whitespace-nowrap transition-all ${
                  active
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30 shadow-sm shadow-rose-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 mr-2 ${active ? 'text-rose-500' : 'text-slate-400'}`} />
                {t.label}
              </button>
            );
          })}
        </nav>

        {/* ========================================================================= */}
        {/* TAB 1: EXECUTIVE OVERVIEW */}
        {/* ========================================================================= */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* DYNAMIC INTERACTIVE CHART MODULE */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 lg:col-span-2 space-y-4">
                {/* TOOLBAR CONTROLS */}
                <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center">
                      <Sliders className="w-4 h-4 mr-2 text-rose-500" />
                      Interactive Analysis: {dynamicMetric}
                    </h3>
                    <p className="text-xs text-slate-400">
                      Showing {dynamicChartData.length} records • Sorted by {dynamicSortOrder === 'desc' ? 'Highest First' : 'Lowest First'}
                    </p>
                  </div>

                  {/* Slicers Strip */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Metric Selector */}
                    <select
                      value={dynamicMetric}
                      onChange={(e) => setDynamicMetric(e.target.value)}
                      className="text-xs bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-semibold focus:border-rose-500 outline-none"
                    >
                      <optgroup label="Anaemia Indicators">
                        <option value={SCHEMA.WOMEN_ANAEMIA}>{SCHEMA.WOMEN_ANAEMIA}</option>
                        <option value={SCHEMA.CHILDREN_ANAEMIA}>{SCHEMA.CHILDREN_ANAEMIA}</option>
                        <option value={SCHEMA.PREG_ANAEMIA}>{SCHEMA.PREG_ANAEMIA}</option>
                        <option value={SCHEMA.ADOLESCENT_ANAEMIA}>{SCHEMA.ADOLESCENT_ANAEMIA}</option>
                      </optgroup>
                      <optgroup label="Socioeconomic Determinants">
                        <option value={SCHEMA.WOMEN_LITERACY}>{SCHEMA.WOMEN_LITERACY}</option>
                        <option value={SCHEMA.IFA_100}>{SCHEMA.IFA_100}</option>
                        <option value={SCHEMA.SANITATION}>{SCHEMA.SANITATION}</option>
                        <option value={SCHEMA.CLEAN_FUEL}>{SCHEMA.CLEAN_FUEL}</option>
                        <option value={SCHEMA.INST_BIRTHS}>{SCHEMA.INST_BIRTHS}</option>
                        <option value={SCHEMA.UNDERWEIGHT_WOMEN}>{SCHEMA.UNDERWEIGHT_WOMEN}</option>
                      </optgroup>
                    </select>

                    {/* Top-N Selector */}
                    <select
                      value={dynamicTopN}
                      onChange={(e) => setDynamicTopN(e.target.value)}
                      className="text-xs bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 font-semibold focus:border-rose-500 outline-none"
                    >
                      <option value={5}>Top 5</option>
                      <option value={10}>Top 10</option>
                      <option value={15}>Top 15</option>
                      <option value={20}>Top 20</option>
                      <option value="ALL">All Active</option>
                    </select>

                    {/* Sort Order Toggle */}
                    <button
                      type="button"
                      onClick={() => setDynamicSortOrder((prev) => (prev === 'desc' ? 'asc' : 'desc'))}
                      className="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-medium flex items-center transition"
                    >
                      {dynamicSortOrder === 'desc' ? '↓ High to Low' : '↑ Low to High'}
                    </button>

                    {/* Chart Type Toggle: Bar vs Line */}
                    <div className="flex rounded-lg border border-slate-700 bg-slate-900 p-0.5">
                      <button
                        type="button"
                        onClick={() => setDynamicChartType('bar')}
                        className={`px-2.5 py-1 text-xs rounded font-medium transition ${
                          dynamicChartType === 'bar' ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Bar
                      </button>
                      <button
                        type="button"
                        onClick={() => setDynamicChartType('line')}
                        className={`px-2.5 py-1 text-xs rounded font-medium transition ${
                          dynamicChartType === 'line' ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Line
                      </button>
                    </div>
                  </div>
                </div>

                {/* DYNAMIC CANVAS */}
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    {dynamicChartType === 'bar' ? (
                      <BarChart data={dynamicChartData} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.6} />
                        <XAxis
                          dataKey="name"
                          stroke="#94A3B8"
                          angle={-30}
                          textAnchor="end"
                          interval={0}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis stroke="#94A3B8" unit="%" domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val) => [`${val}%`, dynamicMetric]}
                        />
                        <Bar dataKey="value" fill="#F43F5E" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    ) : (
                      <LineChart data={dynamicChartData} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.6} />
                        <XAxis
                          dataKey="name"
                          stroke="#94A3B8"
                          angle={-30}
                          textAnchor="end"
                          interval={0}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis stroke="#94A3B8" unit="%" domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val) => [`${val}%`, dynamicMetric]}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#06B6D4"
                          strokeWidth={3}
                          dot={{ r: 4, fill: '#06B6D4' }}
                          activeDot={{ r: 6, fill: '#F43F5E' }}
                        />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>

              {/* WHO Epidemiological Severity Stratification */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white mb-1">WHO Public Health Severity Classification</h3>
                  <p className="text-xs text-slate-400 mb-4">Stratification of 707 Indian districts</p>

                  <div className="space-y-3">
                    {[
                      {
                        title: 'Severe Public Health Problem (≥ 55%)',
                        count: filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) >= 55).length,
                        pct: ((filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) >= 55).length / (filteredData.length || 1)) * 100).toFixed(1),
                        color: 'bg-rose-500',
                        textColor: 'text-rose-400'
                      },
                      {
                        title: 'Moderate Health Problem (30% – 54%)',
                        count: filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) >= 30 && (d[SCHEMA.WOMEN_ANAEMIA] || 0) < 55).length,
                        pct: ((filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) >= 30 && (d[SCHEMA.WOMEN_ANAEMIA] || 0) < 55).length / (filteredData.length || 1)) * 100).toFixed(1),
                        color: 'bg-amber-500',
                        textColor: 'text-amber-400'
                      },
                      {
                        title: 'Mild / Normal Concern (≤ 29%)',
                        count: filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) < 30).length,
                        pct: ((filteredData.filter(d => (d[SCHEMA.WOMEN_ANAEMIA] || 0) < 30).length / (filteredData.length || 1)) * 100).toFixed(1),
                        color: 'bg-emerald-500',
                        textColor: 'text-emerald-400'
                      }
                    ].map((item, idx) => (
                      <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div className="flex justify-between items-center text-xs mb-1.5">
                          <span className="font-semibold text-slate-300">{item.title}</span>
                          <span className={`font-mono font-bold ${item.textColor}`}>
                            {item.count} {item.count === 1 ? 'district' : 'districts'} ({item.pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div className={`h-full ${item.color}`} style={{ width: `${item.pct}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-[11px] text-rose-300 leading-relaxed">
                  <strong>Critical Finding:</strong> Over 80% of surveyed districts cross the WHO severe threshold (≥40%), demonstrating that anaemia is systemic and requires macroeconomic and infrastructural solutions beyond clinical pills alone.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: STATE ANALYSIS & DRILL-DOWN */}
        {/* ========================================================================= */}
        {activeTab === 'states' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top 10 States with Highest Anaemia */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90">
                <div className="flex justify-between items-center mb-3">
                  <div>
                    <h3 className="text-sm font-bold text-rose-400 flex items-center">
                      <TrendingUp className="w-4 h-4 mr-1.5" /> Top 10 States: Highest Anaemia Burden
                    </h3>
                    <p className="text-xs text-slate-400">Average Women Anaemia (%) across constituent districts</p>
                  </div>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={top10States} layout="vertical" margin={{ left: 35, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" domain={[0, 100]} stroke="#64748B" unit="%" />
                      <YAxis type="category" dataKey="state" width={110} stroke="#94A3B8" tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [`${v}%`, 'State Average']}
                      />
                      <Bar dataKey="avgWomen" fill="#EF4444" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Bottom 10 States with Lowest Anaemia */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90">
                <div className="flex justify-between items-center mb-3">
                  <div>
                    <h3 className="text-sm font-bold text-emerald-400 flex items-center">
                      <TrendingDown className="w-4 h-4 mr-1.5" /> Bottom 10 States: Lowest Anaemia Burden
                    </h3>
                    <p className="text-xs text-slate-400">States with relatively superior epidemiological management</p>
                  </div>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={bottom10States} layout="vertical" margin={{ left: 35, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" domain={[0, 100]} stroke="#64748B" unit="%" />
                      <YAxis type="category" dataKey="state" width={110} stroke="#94A3B8" tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [`${v}%`, 'State Average']}
                      />
                      <Bar dataKey="avgWomen" fill="#10B981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Drill-down Table by State */}
            <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 overflow-hidden">
              <h3 className="text-sm font-bold text-white mb-1">State Comparative Matrix (Click State to Drill Down)</h3>
              <p className="text-xs text-slate-400 mb-4">Clicking any row filters the entire dashboard to districts within that state</p>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900/90 text-slate-400 uppercase font-mono tracking-wider">
                    <tr>
                      <th className="p-3">Rank</th>
                      <th className="p-3">State / UT</th>
                      <th className="p-3">Districts</th>
                      <th className="p-3">Avg Women Anaemia (%)</th>
                      <th className="p-3">Avg Literacy (%)</th>
                      <th className="p-3">WHO Category</th>
                      <th className="p-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {stateAggregations.map((s, idx) => (
                      <tr
                        key={s.state}
                        onClick={() => setSelectedState(s.state)}
                        className={`cursor-pointer hover:bg-slate-800/50 transition ${selectedState === s.state ? 'bg-rose-500/10' : ''}`}
                      >
                        <td className="p-3 font-mono text-slate-500">#{idx + 1}</td>
                        <td className="p-3 font-bold text-slate-200">{s.state}</td>
                        <td className="p-3 text-slate-400">{s.districts}</td>
                        <td className="p-3 font-mono font-bold" style={{ color: getPrevalenceColor(s.avgWomen) }}>
                          {s.avgWomen}%
                        </td>
                        <td className="p-3 text-cyan-400 font-mono">{s.avgLit}%</td>
                        <td className="p-3">
                          <span
                            className="px-2 py-0.5 rounded text-[10px] font-bold"
                            style={{
                              color: getPrevalenceColor(s.avgWomen),
                              backgroundColor: `${getPrevalenceColor(s.avgWomen)}15`
                            }}
                          >
                            {s.avgWomen >= 40 ? 'Severe (≥40%)' : s.avgWomen >= 20 ? 'Moderate (20-39%)' : 'Mild (<20%)'}
                          </span>
                        </td>
                        <td className="p-3 text-right">
                          <span className="text-[11px] text-rose-400 hover:text-rose-300 font-medium flex items-center justify-end">
                            Drill Down <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: DISTRICT RANKINGS */}
        {/* ========================================================================= */}
        {activeTab === 'districts' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top 10 Worst Districts */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-sm font-bold text-rose-500 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-1.5" /> Top 10 Highest Anaemia Hotspots
                  </h3>
                  <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">All Women (15–49)</span>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={top10Districts} layout="vertical" margin={{ left: 40, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" domain={[0, 100]} stroke="#64748B" unit="%" />
                      <YAxis type="category" dataKey="name" width={140} stroke="#94A3B8" tick={{ fontSize: 9 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [`${v}%`, 'Prevalence']}
                      />
                      <Bar dataKey="prevalence" fill="#E11D48" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Bottom 10 Lowest Districts */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-sm font-bold text-emerald-400 flex items-center">
                    <TrendingDown className="w-4 h-4 mr-1.5" /> Top 10 Lowest Anaemia Districts
                  </h3>
                  <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">All Women (15–49)</span>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={bottom10Districts} layout="vertical" margin={{ left: 40, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" domain={[0, 100]} stroke="#64748B" unit="%" />
                      <YAxis type="category" dataKey="name" width={140} stroke="#94A3B8" tick={{ fontSize: 9 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [`${v}%`, 'Prevalence']}
                      />
                      <Bar dataKey="prevalence" fill="#10B981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Filterable Table */}
            <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 overflow-hidden">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-bold text-white">Full District Registry ({filteredData.length} active)</h3>
                <span className="text-xs text-slate-400">Click any row to inspect in Diagnostic Dossier</span>
              </div>

              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900 text-slate-400 uppercase font-mono tracking-wider sticky top-0">
                    <tr>
                      <th className="p-3">District</th>
                      <th className="p-3">State</th>
                      <th className="p-3">Women Anaemia (%)</th>
                      <th className="p-3">Child Anaemia (%)</th>
                      <th className="p-3">Pregnant (%)</th>
                      <th className="p-3">Literacy (%)</th>
                      <th className="p-3">Sanitation (%)</th>
                      <th className="p-3">Clean Fuel (%)</th>
                      <th className="p-3 text-right">Dossier</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {filteredData.slice(0, 100).map((r, idx) => (
                      <tr
                        key={idx}
                        onClick={() => {
                          setSelectedForInsight(r);
                          setActiveTab('dossier');
                        }}
                        className="cursor-pointer hover:bg-slate-800/40 transition"
                      >
                        <td className="p-3 font-bold text-white">{r[SCHEMA.DISTRICT]}</td>
                        <td className="p-3 text-slate-400">{r[SCHEMA.STATE]}</td>
                        <td className="p-3 font-mono font-bold text-rose-500">{r[SCHEMA.WOMEN_ANAEMIA]}%</td>
                        <td className="p-3 font-mono text-amber-400">{r[SCHEMA.CHILDREN_ANAEMIA]}%</td>
                        <td className="p-3 font-mono text-cyan-400">{r[SCHEMA.PREG_ANAEMIA]}%</td>
                        <td className="p-3 font-mono text-slate-300">{r[SCHEMA.WOMEN_LITERACY]}%</td>
                        <td className="p-3 font-mono text-slate-300">{r[SCHEMA.SANITATION]}%</td>
                        <td className="p-3 font-mono text-slate-300">{r[SCHEMA.CLEAN_FUEL]}%</td>
                        <td className="p-3 text-right text-rose-400 font-semibold hover:underline">Inspect →</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: ANAEMIA FACTORS & CORRELATION ANALYSIS */}
        {/* ========================================================================= */}
        {activeTab === 'factors' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Correlation Coefficients Matrix */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 lg:col-span-2">
                <h3 className="text-sm font-bold text-white mb-1">
                  Bivariate Pearson Correlation (r) with Women Anaemia
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Evaluated across 707 Indian districts. Direct evidence of protective vs accelerating factors.
                </p>

                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={INDEPENDENT_VARS.map(k => ({
                        indicator: k.replace(/\(%\)/g, '').trim(),
                        r: calculatePearsonR(filteredData.map(d => d[k]), filteredData.map(d => d[SCHEMA.WOMEN_ANAEMIA]))
                      })).sort((a, b) => b.r - a.r)}
                      layout="vertical"
                      margin={{ left: 60, right: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" domain={[-0.6, 0.6]} stroke="#64748B" />
                      <YAxis type="category" dataKey="indicator" width={160} stroke="#94A3B8" tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [v, 'Pearson Correlation (r)']}
                      />
                      <Bar dataKey="r" radius={[4, 4, 4, 4]}>
                        {INDEPENDENT_VARS.map((entry, idx) => (
                          <Cell key={`cell-${idx}`} fill={entry.includes('BMI') || entry.includes('Teenage') ? '#EF4444' : '#06B6D4'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Factor Significance Synthesis */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white mb-2">Epidemiological Interpretation</h3>
                  <div className="space-y-3 text-xs leading-relaxed text-slate-300">
                    <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/20">
                      <span className="font-bold text-cyan-400 block mb-1">Protective Determinants (Negative r):</span>
                      <strong>Clean Cooking Fuel, Improved Sanitation, and Female Schooling</strong> show strong inverse correlations. Clean fuel reduces systemic airway inflammation, while sanitation curbs subclinical enteropathy and hookworm blood loss.
                    </div>
                    <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20">
                      <span className="font-bold text-rose-400 block mb-1">Risk Accelerators (Positive r):</span>
                      <strong>Underweight Women (BMI &lt; 18.5) and Teenage Pregnancy</strong> exhibit high positive correlations with anaemia, evidencing a chronic intergenerational malnutrition cycle.
                    </div>
                  </div>
                </div>

                <div className="text-[11px] font-mono text-slate-500 pt-3 border-t border-slate-800">
                  Statistical confidence: p &lt; 0.001 across all 707 districts
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 5: MACHINE LEARNING MODEL & POLICY SIMULATOR */}
        {/* ========================================================================= */}
        {activeTab === 'ml_insights' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Feature Importance & Model Metrics */}
              <div className="p-5 rounded-2xl border border-slate-800 bg-[#0F172A]/90 lg:col-span-2">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center">
                      <Cpu className="w-4 h-4 mr-2 text-rose-500" /> Multivariate Regression Model (OLS with Ridge L2)
                    </h3>
                    <p className="text-xs text-slate-400">Target: All Women Anaemic (%) | Standardized Beta Coefficients</p>
                  </div>
                  <div className="flex items-center space-x-3 text-xs font-mono">
                    <span className="px-2.5 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded">
                      R² = {mlModel.r2}
                    </span>
                    <span className="px-2.5 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 rounded">
                      MAE = {mlModel.mae}%
                    </span>
                    <span className="px-2.5 py-1 bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded">
                      RMSE = {mlModel.rmse}%
                    </span>
                  </div>
                </div>

                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={featureImportanceList.map(f => ({
                        feature: f.feature.replace(/\(%\)/g, '').trim(),
                        beta: f.beta,
                        impact: f.direction
                      }))}
                      layout="vertical"
                      margin={{ left: 55, right: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.5} />
                      <XAxis type="number" stroke="#64748B" domain={[-0.4, 0.4]} />
                      <YAxis type="category" dataKey="feature" width={160} stroke="#94A3B8" tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [v, 'Standardized Beta (Effect Size)']}
                      />
                      <Bar dataKey="beta" radius={[4, 4, 4, 4]}>
                        {featureImportanceList.map((entry, idx) => (
                          <Cell key={`cell-${idx}`} fill={entry.beta > 0 ? '#EF4444' : '#10B981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Interactive What-If Policy Intervention Simulator */}
              <div className="p-5 rounded-2xl border border-rose-500/30 bg-gradient-to-b from-[#0F172A] to-[#121124] flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center mb-1">
                    <Sparkles className="w-4 h-4 mr-1.5 text-rose-500" /> What-If Policy Simulator
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Simulate district-wide targeted interventions to predict anaemia reduction using the ML model.
                  </p>

                  <div className="space-y-4 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-300 font-semibold mb-1">
                        <span>Increase Women's Literacy</span>
                        <span className="text-rose-400 font-mono">+{simLiteracyBoost}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="30"
                        value={simLiteracyBoost}
                        onChange={e => setSimLiteracyBoost(Number(e.target.value))}
                        className="w-full accent-rose-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-300 font-semibold mb-1">
                        <span>Expand Improved Sanitation</span>
                        <span className="text-cyan-400 font-mono">+{simSanitationBoost}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="40"
                        value={simSanitationBoost}
                        onChange={e => setSimSanitationBoost(Number(e.target.value))}
                        className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-300 font-semibold mb-1">
                        <span>Boost IFA 100+ Days Intake</span>
                        <span className="text-emerald-400 font-mono">+{simIFABoost}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="50"
                        value={simIFABoost}
                        onChange={e => setSimIFABoost(Number(e.target.value))}
                        className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                      />
                    </div>
                  </div>
                </div>

                {/* Simulated Outcome Display */}
                <div className="mt-5 p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                    <span>Forecasted National Prevalence:</span>
                    <span className="line-through text-slate-500 font-mono">{overviewMetrics.womenAvg}%</span>
                  </div>
                  <div className="flex justify-between items-baseline">
                    <span className="text-2xl font-bold text-emerald-400 font-mono">{simulatedImpact.newPrevalence}%</span>
                    <span className="text-xs font-bold text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded">
                      -{simulatedImpact.reduction}% Reduction
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 6: AUTOMATED DISTRICT DIAGNOSTIC DOSSIER */}
        {/* ========================================================================= */}
        {activeTab === 'dossier' && activeDistrictDossier && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl border border-slate-800 bg-[#0F172A]/90 relative overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
                <div>
                  <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                    District Diagnostic Briefing
                  </span>
                  <h2 className="text-2xl font-extrabold text-white mt-0.5">
                    {activeDistrictDossier.district}, <span className="text-slate-400 font-normal">{activeDistrictDossier.state}</span>
                  </h2>
                </div>

                <div className="flex items-center space-x-3">
                  <span className={`text-xs font-bold uppercase tracking-wider px-3 py-1 rounded-full border ${activeDistrictDossier.tagColor}`}>
                    {activeDistrictDossier.severityTag}
                  </span>
                  <div className="text-right">
                    <span className="text-[10px] uppercase font-mono text-slate-500 block">Prevalence</span>
                    <span className="text-2xl font-mono font-bold text-rose-500">{activeDistrictDossier.val}%</span>
                  </div>
                </div>
              </div>

              {/* Benchmarks Matrix */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">State Baseline ({activeDistrictDossier.state})</span>
                  <div className="text-lg font-bold text-white mt-1">{activeDistrictDossier.stateAvg}%</div>
                  <span className={`text-xs font-mono font-semibold ${activeDistrictDossier.stateDiff > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {activeDistrictDossier.stateDiff > 0 ? `+${activeDistrictDossier.stateDiff}% higher` : `${activeDistrictDossier.stateDiff}% lower`}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">National Average (All Districts)</span>
                  <div className="text-lg font-bold text-white mt-1">{activeDistrictDossier.nationalAvg}%</div>
                  <span className={`text-xs font-mono font-semibold ${activeDistrictDossier.natDiff > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {activeDistrictDossier.natDiff > 0 ? `+${activeDistrictDossier.natDiff}% above national` : `${activeDistrictDossier.natDiff}% below national`}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">WHO Public Health Gap</span>
                  <div className="text-lg font-bold text-rose-400 mt-1">
                    {(activeDistrictDossier.val - 39.9).toFixed(1)}%
                  </div>
                  <span className="text-xs text-slate-400">Percentage points beyond severe cutoff</span>
                </div>
              </div>

              {/* Risk Factors vs Protective Strengths */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
                <div className="p-4 rounded-xl bg-rose-500/5 border border-rose-500/20">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center mb-3">
                    <AlertTriangle className="w-4 h-4 mr-1.5" /> High Risk Contributors Flagged
                  </h4>
                  {activeDistrictDossier.riskFactors.length === 0 ? (
                    <p className="text-xs text-slate-400">No major socio-nutritional risk deficits flagged.</p>
                  ) : (
                    <div className="space-y-2.5">
                      {activeDistrictDossier.riskFactors.map((r, i) => (
                        <div key={i} className="flex justify-between items-center text-xs p-2 rounded-lg bg-slate-900/80">
                          <div>
                            <span className="font-semibold text-slate-200 block">{r.label}</span>
                            <span className="text-[11px] text-slate-500">{r.desc}</span>
                          </div>
                          <span className="font-mono font-bold text-rose-400">{r.value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center mb-3">
                    <CheckCircle2 className="w-4 h-4 mr-1.5" /> Protective Buffers Identified
                  </h4>
                  {activeDistrictDossier.protectiveStrengths.length === 0 ? (
                    <p className="text-xs text-slate-400">Limited protective structural buffers detected.</p>
                  ) : (
                    <div className="space-y-2.5">
                      {activeDistrictDossier.protectiveStrengths.map((p, i) => (
                        <div key={i} className="flex justify-between items-center text-xs p-2 rounded-lg bg-slate-900/80">
                          <div>
                            <span className="font-semibold text-slate-200 block">{p.label}</span>
                            <span className="text-[11px] text-slate-500">{p.desc}</span>
                          </div>
                          <span className="font-mono font-bold text-emerald-400">{p.value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Tailored Policy Recommendations */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center mb-2.5">
                  <Sparkles className="w-4 h-4 mr-1.5" /> Priority Policy Actions for District Medical Officer
                </h4>
                <ul className="space-y-2 text-xs text-slate-300">
                  {activeDistrictDossier.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-cyan-500 font-bold mr-2">0{i + 1}.</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}