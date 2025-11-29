import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Typography, Card, Space, Tag, Button, Alert, Spin, message } from 'antd';
import { ReloadOutlined, ArrowLeftOutlined, DownCircleOutlined } from '@ant-design/icons';
import RunResults from '../components/RunResults';
import { runsApi } from '../services/api';

const { Title, Text } = Typography;

function RunMonitoring() {
    const { runId } = useParams();
    const navigate = useNavigate();
    const [run, setRun] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const resultsRef = useRef(null); // Ref for scrolling to results
    const storedCountsRef = useRef(false); // Avoid duplicate final-counts fetch

    // Scroll to results function
    const scrollToResults = () => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
    };
    
    useEffect(() => {
        let isMounted = true;
        let intervalId = null;

        const fetchRunDetails = async () => {
            if (!isMounted) return;
            try {
                const response = await fetch(`/api/runs/${runId}`);
                if (!response.ok) throw new Error('Failed to fetch run details');
                const runData = await response.json();
                
                if (isMounted) {
                    setRun(runData);
                    setError(null);
                    // If the run is no longer running, stop the interval
                    if (runData.status !== 'running' && runData.status !== 'pending') {
                        clearInterval(intervalId);
                    }
                }
            } catch (err) {
                if (isMounted) {
                    setError(err.message);
                    clearInterval(intervalId);
                }
            } finally {
                if (isMounted) {
                    setIsLoading(false);
                    setIsRefreshing(false);
                }
            }
        };

        setIsLoading(true);
        fetchRunDetails(); // Initial fetch

        // Set up interval for polling
        intervalId = setInterval(fetchRunDetails, 10000); // Poll every 10 seconds

        // Cleanup function
        return () => {
            isMounted = false;
            clearInterval(intervalId);
        };
    }, [runId]);

    const refreshRunStatus = async () => {
        setIsRefreshing(true);
        // The useEffect hook will handle the actual fetching
    };

    // Fetch final counts once run completes
    useEffect(() => {
        const fetchFinalCounts = async () => {
            try {
                const existing = run?.final_result_json?.counts;
                if (existing) {
                    const next = {
                        cloud: Number(existing.cloud ?? 1),
                        edge: Number(existing.edge ?? 1),
                        device: Number(existing.device ?? 1),
                    };
                    localStorage.setItem('preciseCounts', JSON.stringify(next));
                    message.success('Precise counts saved. View and adjust on the Planner page.');
                    storedCountsRef.current = true;
                    return;
                }

                const resp = await runsApi.getResults(runId);
                const payload = resp.data;
                const c = payload?.counts;
                if (c) {
                    const next = {
                        cloud: Number(c.cloud ?? 1),
                        edge: Number(c.edge ?? 1),
                        device: Number(c.device ?? 1),
                    };
                    localStorage.setItem('preciseCounts', JSON.stringify(next));
                    message.success('Precise counts saved. View and adjust on the Planner page.');
                    storedCountsRef.current = true;
                    return;
                }

                const resp2 = await fetch(`/api/runs/${runId}/final-counts`);
                if (!resp2.ok) {
                    const text = await resp2.text();
                    let detail = '';
                    try { detail = JSON.parse(text)?.detail; } catch {}
                    throw new Error(detail || 'Failed to fetch final counts');
                }
                const data = await resp2.json();
                const next2 = {
                    cloud: Number(data.cloud ?? 1),
                    edge: Number(data.edge ?? 1),
                    device: Number(data.device ?? 1),
                };
                localStorage.setItem('preciseCounts', JSON.stringify(next2));
                    message.success('Precise counts saved. View and adjust on the Planner page.');
                storedCountsRef.current = true;
            } catch (err) {
                const msg = err?.message || 'Failed to fetch final counts';
                message.error(`Failed to fetch final counts: ${msg}`);
            }
        };

        if (run && run.status === 'completed' && !storedCountsRef.current) {
            fetchFinalCounts();
        }
    }, [run, runId, navigate]);

    if (isLoading) {
        return <div style={{ textAlign: 'center', padding: '50px 0' }}><Spin size="large" /></div>;
    }

    if (error) {
        return <Alert message="Error" description={error} type="error" showIcon />;
    }

    if (!run) {
        return <Alert message="Run Not Found" description="The specified run could not be found." type="warning" />;
    }

    const getStatusColor = (status) => ({
        pending: 'default',
        running: 'processing',
        completed: 'success',
        failed: 'error',
    }[status] || 'default');

    const renderStatusContent = () => {
        switch (run.status) {
            case 'completed':
                return (
                    <div ref={resultsRef} style={{ paddingTop: 24 }}>
                        <RunResults run={run} />
                    </div>
                );
            case 'failed':
                return (
                    <Card>
                        <Alert message="Optimization Failed" description="Check the logs for details." type="error" showIcon />
                        {run.logs && (
                            <div style={{ marginTop: 16 }}>
                                <Title level={4}>Error Logs</Title>
                                <pre style={{ backgroundColor: '#fff2f0', padding: 16, borderRadius: 6, overflow: 'auto', maxHeight: 300, border: '1px solid #ffccc7' }}>
                                    {run.logs}
                                </pre>
                            </div>
                        )}
                    </Card>
                );
            case 'running':
                return (
                    <Card>
                        <Alert message="Optimization in Progress" description="This page automatically refreshes." type="info" showIcon />
                        <div style={{ marginTop: 16, textAlign: 'center' }}>
                            <Spin size="large" tip="Processing..." />
                        </div>
                    </Card>
                );
            case 'pending':
                 return (
                    <Card>
                        <Alert message="Run Pending" description="Waiting for the executor service to start." type="warning" showIcon />
                    </Card>
                );
            default:
                return null;
        }
    };

    return (
        <div>
            <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Link to="/">
                    <Button icon={<ArrowLeftOutlined />}>Back to Dashboard</Button>
                </Link>
                <Space>
                    {run.status === 'completed' && (
                        <Button type="primary" icon={<DownCircleOutlined />} onClick={scrollToResults}>
                            View Results
                        </Button>
                    )}
                     <Button icon={<ReloadOutlined />} onClick={refreshRunStatus} loading={isRefreshing}>
                        Refresh
                    </Button>
                </Space>
            </div>

            <Card style={{ marginBottom: 24 }}>
                <Title level={2}>Run Monitoring</Title>
                <Space direction="vertical" size="middle">
                    <Text strong>Run ID: <Text code copyable>{runId}</Text></Text>
                    <Text strong>Status: <Tag color={getStatusColor(run.status)}>{run.status.toUpperCase()}</Tag></Text>
                    {run.start_time && <Text>Started: {new Date(run.start_time).toLocaleString()}</Text>}
                    {run.end_time && <Text>Completed: {new Date(run.end_time).toLocaleString()}</Text>}
                </Space>
            </Card>

            {renderStatusContent()}
        </div>
    );
}

export default RunMonitoring;
