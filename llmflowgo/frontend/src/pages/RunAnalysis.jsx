import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Card, Typography, Empty, Spin, Alert } from 'antd';
import ReactECharts from 'echarts-for-react';
import { runsApi } from '../services/api';

const { Title, Paragraph } = Typography;

const isNumberArray = (arr) => Array.isArray(arr) && arr.length > 0 && arr.every(v => typeof v === 'number' && Number.isFinite(v));

const RunAnalysis = () => {
  const { runId } = useParams();
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const resp = await runsApi.getById(runId);
        const data = resp.data;
        setRun(data || null);
        setError(null);
      } catch (e) {
        console.error(e);
        setError(e?.response?.data?.detail || 'Failed to fetch run results');
      } finally {
        setLoading(false);
      }
    };
    if (runId) fetchData();
  }, [runId]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '40px 0' }}><Spin size="large" /></div>;
  }

  if (error) {
    return <Alert type="error" message="Error" description={error} showIcon />;
  }

  if (!run || run.status !== 'completed') {
    return (
      <Card>
        <Empty description={<span>No analysis available (run incomplete or results unavailable)</span>} />
      </Card>
    );
  }

  // Find the best candidate solution (preferring best_solution_code in final_population and pareto_front)
  const candidates = [
    ...Array.isArray(run.final_population) ? run.final_population : [],
    ...Array.isArray(run.pareto_front) ? run.pareto_front : [],
  ];
  const bestSolution = candidates.find(sol => sol.code === run.best_solution_code) || candidates[0];
  const baselineList = Array.isArray(run?.baseline_scores) ? run.baseline_scores : null;
  const objectiveNamesAll = run?.final_result_json?.meta?.objective_names;
  const lowerNames = Array.isArray(objectiveNamesAll) ? objectiveNamesAll.map(n => String(n).toLowerCase()) : [];
  const idxMakespan = lowerNames.indexOf('makespan');
  const idxEnergy = lowerNames.indexOf('energy');
  const idxCost = lowerNames.indexOf('cost');
  const indicesValid = idxMakespan > 2 && idxEnergy > 2 && idxCost > 2;
  const bestCandidate = Array.isArray(run?.pareto_front) && run?.pareto_front.length > 0
    ? (run.pareto_front.find(sol => sol.code === run.best_solution_code) || run.pareto_front[0])
    : null;
  const bestObj = bestCandidate && Array.isArray(bestCandidate.objectives) ? bestCandidate.objectives : null;
  const algoLabelsBase = baselineList ? baselineList.map(b => String(b?.name || '')) : [];
  const algoLabels = bestObj ? [...algoLabelsBase, 'Best'] : algoLabelsBase;
  const canShowChart = baselineList && baselineList.length > 0 && indicesValid && algoLabels.every(l => l.length > 0);

  const buildMetricData = (idx) => {
    const arr = baselineList.map(b => {
      const scores = Array.isArray(b?.scores) ? b.scores : [];
      const v = (idx >= 0 && idx < scores.length) ? scores[idx] : null;
      return typeof v === 'number' ? Math.abs(v) : null;
    });
    if (bestObj && idx >= 0 && idx < bestObj.length) {
      arr.push(Math.abs(bestObj[idx]));
    }
    return arr;
  };

  const series = canShowChart ? [
    { name: 'Time', type: 'bar', itemStyle: { color: '#EEC0A0' }, data: buildMetricData(idxMakespan) },
    { name: 'Energy', type: 'bar', itemStyle: { color: '#1f2b3d' }, data: buildMetricData(idxEnergy) },
    { name: 'Cost', type: 'bar', itemStyle: { color: '#8B0000' }, data: buildMetricData(idxCost) },
  ] : [];

  const analysisChartOption = canShowChart ? {
    title: { text: 'Algorithm vs Best Solution Comparison', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['Time', 'Energy', 'Cost'], top: 24 },
    grid: { left: '3%', right: '3%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: algoLabels, axisLabel: { fontWeight: 'bold', fontSize: 14 } },
    yAxis: { type: 'value', axisLabel: { fontSize: 14 } },
    series
  } : null;

  return (
    <Card>
      <Title level={3}>Final Results</Title>

      {analysisChartOption ? (
        <>
          <Paragraph>
            Chart shows baseline vs best solution metrics (aligned by objectives).
          </Paragraph>
          <ReactECharts option={analysisChartOption} notMerge={true} lazyUpdate={true} style={{ height: 480 }} />
        </>
      ) : (
        <Paragraph type="secondary">
          Unable to generate analysis chart: missing baseline or best candidate, or insufficient objective dimensions.
        </Paragraph>
      )}

    </Card>
  );
};

export default RunAnalysis;
