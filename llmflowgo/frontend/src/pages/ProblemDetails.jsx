import React, { useState, useEffect, useContext } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Typography, Spin, Alert, Descriptions, Button, Card, Form, Input, message, Table, Radio, Tag, Space, Popconfirm, Tooltip } from 'antd';
import { ArrowLeftOutlined, ExperimentOutlined, BarChartOutlined, EyeOutlined, DeleteOutlined, SettingOutlined } from '@ant-design/icons';
import { problemsApi } from '../services/api';
import useLLMConfigStore from '../store/llmConfigStore';
import { AppContext } from '../App'; 

const { Title, Text, Paragraph } = Typography;

function ProblemDetails() {
    const { problemId } = useParams();
    const [problem, setProblem] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [selectedFunction, setSelectedFunction] = useState(null);
    const [runs, setRuns] = useState([]);
    const [isLoadingRuns, setIsLoadingRuns] = useState(false);
    const [form] = Form.useForm();
    
    // --- ZUSTAND FIX: Correctly select state from the store ---
    const { apiKey, baseUrl, modelName } = useLLMConfigStore();
    const llmConfig = { apiKey, baseUrl, modelName }; // Re-create the object for convenience
    const { setIsModalOpen } = useContext(AppContext);

    // --- HYDRATION FIX START ---
    // This ensures that we wait for the client to mount and the store to hydrate from localStorage
    // before checking the configuration status.
    const [isClient, setIsClient] = useState(false);
    useEffect(() => {
        setIsClient(true);
    }, []);
    // --- HYDRATION FIX END ---

    // Now, this check will only be truly evaluated after the component has mounted on the client.
    const isLLMConfigured = isClient && llmConfig.apiKey && llmConfig.baseUrl && llmConfig.modelName;

    // Delete running history
    const handleDeleteRun = async (runId) => {
        try {
            const response = await fetch(`/api/runs/${runId}`, {
                method: 'DELETE',
            });
            
            if (response.ok) {
                message.success('Run deleted successfully');
                // Remove the deleted run from local state
                setRuns(runs.filter(run => run.id !== runId));
            } else {
                throw new Error('Failed to delete run');
            }
        } catch (err) {
            message.error('Failed to delete run: ' + err.message);
        }
    };

    useEffect(() => {
        const fetchDetails = async () => {
            try {
                setIsLoading(true);
                const response = await problemsApi.getById(problemId);
                const problemData = response.data;
                setProblem(problemData);
                
                if (problemData.target_function_name) {
                    setSelectedFunction(problemData.target_function_name);
                    form.setFieldsValue({
                        target_function_name: problemData.target_function_name,
                        task_description: problemData.task_description,
                    });
                }
            } catch (err) {
                setError('Failed to load problem details.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchDetails();
    }, [problemId, form]);

    // Get running history of this problem package
    useEffect(() => {
        const fetchRuns = async () => {
            try {
                setIsLoadingRuns(true);
                const response = await fetch(`/api/runs?problem_id=${problemId}`);
                if (response.ok) {
                    const runsData = await response.json();
                    setRuns(runsData);
                }
            } catch (err) {
                console.error('Failed to fetch runs:', err);
            } finally {
                setIsLoadingRuns(false);
            }
        };
        
        if (problemId) {
            fetchRuns();
        }
    }, [problemId]);

    const handleAnalyze = async () => {
        // --- Critical Diagnostic Log ---
        console.log('--- Analyzing Code Button Clicked ---');
        console.log('Is client mounted?', isClient);
        console.log('LLM Config from store:', llmConfig);
        console.log('Is LLM configured?', isLLMConfigured);
        console.log('--- End of Diagnostics ---');

        if (!isLLMConfigured) {
            message.warning("Please configure your LLM settings using the 'LLM Settings' button in the top navigation bar.");
            setIsModalOpen(true); // Still open it for convenience
            return;
        }

        setIsAnalyzing(true);
        setAnalysisResult(null);
        setSelectedFunction(null);
        form.resetFields(['target_function_name']);
        try {
            const response = await problemsApi.analyze(problemId, llmConfig);
            setAnalysisResult(response.data.analysis);
            message.success('Analysis complete! Please select a target function.');
        } catch (err) {
            const errorDetail = err.response?.data?.detail || 'Failed to analyze the code.';
            message.error(errorDetail);
        } finally {
            setIsAnalyzing(false);
        }
    };
    
    const onFinish = async (values) => {
        if (!selectedFunction) {
            message.error('Please select a target function from the table.');
            return;
        }
        
        const configData = {
            target_function_name: selectedFunction,
            task_description: values.task_description,
            llm_config: llmConfig // Pass LLM config during final configuration
        };
        
        setIsSaving(true);
        try {
            const response = await problemsApi.configure(problemId, configData);
            setProblem(response.data);
            message.success('Configuration saved successfully!');
        } catch (err) {
            message.error('Failed to save configuration.');
        } finally {
            setIsSaving(false);
        }
    };
    
    const columns = [
        {
            title: 'Select',
            key: 'select',
            render: (_, record) => (
                <Radio
                    checked={selectedFunction === record.name}
                    onChange={() => setSelectedFunction(record.name)}
                />
            ),
        },
        { title: 'Function Name', dataIndex: 'name', key: 'name' },
        { title: 'Optimization Potential', dataIndex: 'potential', key: 'potential', sorter: (a, b) => b.potential - a.potential },
        { title: 'Reason from LLM', dataIndex: 'reason', key: 'reason' },
    ];

    // Running history table column definition
    const runColumns = [
        {
            title: 'Run ID',
            dataIndex: 'id',
            key: 'id',
            width: 280,
            render: (id) => <Text code copyable>{id}</Text>,
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 120,
            render: (status) => {
                const statusColors = {
                    'pending': 'default',
                    'running': 'processing',
                    'completed': 'success',
                    'failed': 'error'
                };
                return <Tag color={statusColors[status] || 'default'}>{status.toUpperCase()}</Tag>;
            },
        },
        {
            title: 'Start Time',
            dataIndex: 'start_time',
            key: 'start_time',
            width: 180,
            render: (startTime) => startTime ? new Date(startTime).toLocaleString() : '-',
        },
        {
            title: 'End Time',
            dataIndex: 'end_time',
            key: 'end_time',
            width: 180,
            render: (endTime) => endTime ? new Date(endTime).toLocaleString() : '-',
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 180,
            render: (_, record) => (
                <Space>
                    <Button
                        type="primary"
                        size="small"
                        icon={<EyeOutlined />}
                        onClick={() => window.open(`/run/${record.id}`, '_blank')}
                    >
                        View Run
                    </Button>
                    <Popconfirm
                        title="Delete this run?"
                        description="Are you sure you want to delete this run? This action cannot be undone."
                        onConfirm={() => handleDeleteRun(record.id)}
                        okText="Yes"
                        cancelText="No"
                        okType="danger"
                    >
                        <Button
                            type="primary"
                            danger
                            size="small"
                            icon={<DeleteOutlined />}
                        >
                            Delete
                        </Button>
                    </Popconfirm>
                </Space>
            ),
        },
    ];

    if (isLoading) {
        return <div style={{ textAlign: 'center', padding: '50px 0' }}><Spin size="large" /></div>;
    }

    if (error) {
        return <Alert message="Error" description={error} type="error" showIcon />;
    }

    if (!problem) {
        return <Alert message="Not Found" description="This edge workflow does not exist." type="warning" showIcon />;
    }

    return (
        <div>
            <Button style={{ marginBottom: 16 }}>
                <Link to="/"><ArrowLeftOutlined /> Back to Dashboard</Link>
            </Button>
            <Card style={{ marginBottom: 24 }}>
                <Title level={2}>{problem.name}</Title>
                <Descriptions bordered column={1}>
                    <Descriptions.Item label="Description">{problem.description}</Descriptions.Item>
                    <Descriptions.Item label="Edge Workflow ID"><Text copyable>{problem.id}</Text></Descriptions.Item>
                    <Descriptions.Item label="Framework File Path">{problem.framework_file_path}</Descriptions.Item>
                    <Descriptions.Item label="Data Directory Path">{problem.data_file_path}</Descriptions.Item>
                    <Descriptions.Item label="Get Instance File Path">{problem.get_instance_file_path}</Descriptions.Item>
                </Descriptions>
            </Card>

            <Card style={{ marginTop: 24 }}>
                <Title level={3}>Optimization Configuration</Title>
                
                <Space style={{ marginBottom: 24 }}>
                    <Button
                        type="primary"
                        icon={<ExperimentOutlined />}
                        onClick={handleAnalyze}
                        loading={isAnalyzing}
                    >
                        {problem?.target_function_name ? 'Re-Analyze Code' : 'Analyze Code'}
                    </Button>
                    {isLLMConfigured ? (
                        <Tooltip title={`Using model: ${llmConfig.modelName}`}>
                            <Tag color="success">LLM Configured</Tag>
                        </Tooltip>
                    ) : (
                        <Tooltip title="Please use the 'LLM Settings' button in the top navigation bar to configure.">
                            <Tag color="warning">LLM Not Configured</Tag>
                        </Tooltip>
                    )}
                </Space>

                {(analysisResult || problem?.target_function_name) && (
                     <Form form={form} layout="vertical" onFinish={onFinish}>
                        <Paragraph>
                            Based on the analysis, select one function to be the optimization target for the MEOH framework.
                        </Paragraph>
                        
                        {analysisResult && (
                             <Table
                                rowKey="name"
                                columns={columns}
                                dataSource={analysisResult}
                                pagination={false}
                                style={{ marginBottom: 24 }}
                            />
                        )}
                        
                        {selectedFunction && (
                            <>
                                <Alert
                                    message={`You have selected "${selectedFunction}" as the target function.`}
                                    type="info"
                                    showIcon
                                    style={{ marginBottom: 24 }}
                                />
                                <Form.Item
                                    name="task_description"
                                    label="Task Description for LLM"
                                    rules={[{ required: true }]}
                                >
                                    <Input.TextArea rows={4} placeholder={`Describe the goal for improving the "${selectedFunction}" function...`} />
                                </Form.Item>
                                <Form.Item>
                                    <Button type="primary" htmlType="submit" loading={isSaving}>
                                        Save Configuration
                                    </Button>
                                </Form.Item>
                            </>
                        )}
                    </Form>
                )}
            </Card>

            {/* Run History Card */}
            <Card id="results" style={{ marginTop: 24 }}>
                <Title level={3}>
                    <BarChartOutlined /> Optimization Results History
                </Title>
                
                {isLoadingRuns ? (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <Spin size="large" />
                        <div style={{ marginTop: 8 }}>Loading run history...</div>
                    </div>
                ) : runs.length > 0 ? (
                    <div>
                        <Paragraph>
                            This edge workflow has {runs.length} optimization run(s). Click "View Run" to see detailed results for each execution.
                        </Paragraph>
                        <Table
                            rowKey="id"
                            columns={runColumns}
                            dataSource={runs}
                            pagination={{
                                pageSize: 10,
                                showSizeChanger: true,
                                showQuickJumper: true,
                                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} runs`
                            }}
                            scroll={{ x: 1000 }}
                        />
                    </div>
                ) : (
                    <Alert
                        message="No Runs Yet"
                        description="This edge workflow hasn't been run for optimization yet. Start a new run to see results here."
                        type="info"
                        showIcon
                    />
                )}
            </Card>

        </div>
    );
}

export default ProblemDetails;
