import React, { useState, useEffect, useRef } from 'react';
import { Typography, Button, Space, Card, List, Spin, Alert, Modal, Form, Input, message, Tag, Tooltip, Popconfirm } from 'antd';
import { PlayCircleOutlined, EyeOutlined, BarChartOutlined, DeleteOutlined } from '@ant-design/icons';
import useProblemStore from '../store/problemStore';
import { useNavigate } from 'react-router-dom';

const { Title, Paragraph, Text } = Typography;

function Dashboard() {
    const { problems, isLoading, error, fetchProblems, deleteProblem } = useProblemStore();
    const [runModalVisible, setRunModalVisible] = useState(false);
    const [selectedProblem, setSelectedProblem] = useState(null);
    const [runForm] = Form.useForm();
    const navigate = useNavigate();
    const autoRunSilentRef = useRef(false);

    useEffect(() => {
        fetchProblems();
    }, [fetchProblems]);

    useEffect(() => {
        // Diagnostic log, check the data retrieved from the store
        console.log("Fetched problems from store:", problems);
    }, [problems]);

    useEffect(() => {
      try {
        const raw = localStorage.getItem('autoRunPayload');
        if (!raw) return;
        const payload = JSON.parse(raw);
        if (!payload || typeof payload !== 'object') return;

        // If a runId exists, it means a Run has already been created in the one-click process, so go directly.
        if (payload.runId) {
          try { localStorage.removeItem('autoRunPayload'); } catch {}
          // navigate(`/run/${payload.runId}`);
          return;
        }

        const targetProblemId = payload.problemId;
        const incomingConfig = payload.config || {};
        if (!targetProblemId) return;

        const target = problems.find(p => p.id === targetProblemId);
        if (!target) return;

        // Set the currently selected question so that the problem_id can be obtained when submitting
        setSelectedProblem(target);

        // Flatten the MEOH parameters into the form and merge them with the incoming configuration
        runForm.setFieldsValue({
          pop_size: 20,
          max_generations: 10,
          max_sample_nums: 100,
          selection_num: 4,
          use_e2_operator: true,
          use_m1_operator: true,
          use_m2_operator: true,
          num_samplers: 4,
          num_evaluators: 4,
          ...incomingConfig,
        });

        // Automatically submit the Run (trigger the form's onFinish)
        autoRunSilentRef.current = true;
        runForm.submit();

        // Clean up the autoRunPayload to avoid duplicate triggers
        try { localStorage.removeItem('autoRunPayload'); } catch {}
      } catch (e) {
        console.warn('Failed to process autoRunPayload', e);
      }
    }, [problems, navigate, runForm]);

    const handleRunClick = (problem) => {
        setSelectedProblem(problem);
        setRunModalVisible(true);
    };

    const handleDelete = async (problem) => {
        try {
            await deleteProblem(problem.id);
            message.success('Deleted successfully');
        } catch (err) {
            message.error(err?.message || 'Delete failed');
        }
    };

    const handleRunSubmit = async (values) => {
        try {
            const response = await fetch('/api/runs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    problem_id: selectedProblem.id,
                    meoh_config: {
                        pop_size: parseInt(values.pop_size),
                        max_generations: parseInt(values.max_generations),
                        max_sample_nums: parseInt(values.max_sample_nums),
                        selection_num: parseInt(values.selection_num),
                        use_e2_operator: values.use_e2_operator !== false,
                        use_m1_operator: values.use_m1_operator !== false,
                        use_m2_operator: values.use_m2_operator !== false,
                        num_samplers: parseInt(values.num_samplers),
                        num_evaluators: parseInt(values.num_evaluators),
                    }
                }),
            });

            if (response.ok) {
                const runData = await response.json();
                message.success('Run started successfully!');
                setRunModalVisible(false);
                runForm.resetFields();
                if (!autoRunSilentRef.current) {
                    navigate(`/run/${runData.id}`);
                } else {
                    autoRunSilentRef.current = false;
                }
            } else {
                throw new Error('Failed to start run');
            }
        } catch (error) {
            message.error('Failed to start run: ' + error.message);
        }
    };

    const getStatusColor = (status) => {
        const colors = {
            pending: 'default',
            running: 'processing',
            completed: 'success',
            failed: 'error',
        };
        return colors[status] || 'default';
    };

    return (
        <div>
            <Card style={{ marginBottom: 24 }}>
                <Title>ISEC LABORATORY</Title>
                <Paragraph>
                    This platform allows you to define edge workflows, leverage LLMs to evolve heuristics, and find optimal solutions.
                </Paragraph>
                <Space size="middle">
                    <Button type="primary" size="large" onClick={() => window.location.href = '/workflow'}>
                        Define a New Edge Workflow
                    </Button>
                    <Button size="large" onClick={() => window.open('http://localhost:5174/workflow', '_blank')}>
                        Create a New Workflow
                    </Button>
                </Space>
            </Card>

            <Title level={2}>Existing Edge Workflows</Title>
            {isLoading ? (
                <div style={{ textAlign: 'center', padding: '50px 0' }}><Spin size="large" /></div>
            ) : error ? (
                <Alert message="Error" description={error} type="error" showIcon />
            ) : (
                <List
                    grid={{ gutter: 16, xs: 1, sm: 2, md: 3, lg: 3, xl: 4, xxl: 4 }}
                    dataSource={Array.isArray(problems) ? problems : []}
                    renderItem={(item) => (
                        <List.Item>
                            <Card
                                title={item.name}
                                bodyStyle={{ minHeight: 200, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
                                actions={[
                                    <Tooltip title={!item.llm_config ? "Please configure LLM settings on the Details page first" : ""}>
                                        <Button 
                                            key="run" 
                                            type="primary" 
                                            icon={<PlayCircleOutlined />} 
                                            onClick={() => handleRunClick(item)}
                                            disabled={!item.llm_config}
                                        >
                                            RUN
                                        </Button>
                                    </Tooltip>,
                                    <Button key="details" icon={<EyeOutlined />} onClick={() => window.location.href = `/problem/${item.id}`}>DETAILS</Button>,
                                    <Button key="results" danger icon={<BarChartOutlined />} onClick={() => window.location.href = `/problem/${item.id}#results`}>RESULTS</Button>,
                                    <Popconfirm
                                        key="delete-pop"
                                        title="Confirm delete this Edge Workflow?"
                                        okText="Delete"
                                        cancelText="Cancel"
                                        onConfirm={() => handleDelete(item)}
                                    >
                                        <Button key="delete" danger icon={<DeleteOutlined />}>Delete</Button>
                                    </Popconfirm>,
                                ]}
                            >
                                <Card.Meta
                                    description={item.description}
                                />
                                {!item.llm_config && (
                                    <Tag color="orange" style={{ marginTop: 10 }}>
                                        Configuration Pending
                                    </Tag>
                                )}
                            </Card>
                        </List.Item>
                    )}
                />
            )}

            <Modal
                title={`Run Optimization: ${selectedProblem?.name}`}
                open={runModalVisible}
                onCancel={() => setRunModalVisible(false)}
                footer={null}
                width={600}
            >
                <Form
                    form={runForm}
                    layout="vertical"
                    onFinish={handleRunSubmit}
                    initialValues={{
                        pop_size: 20, max_generations: 10, max_sample_nums: 100,
                        selection_num: 4, use_e2_operator: true, use_m1_operator: true,
                        use_m2_operator: true, num_samplers: 4, num_evaluators: 4,
                    }}
                >
                    {/* Form Items */}
                    <Form.Item name="pop_size" label="Population Size" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item name="max_generations" label="Max Generations" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item name="max_sample_nums" label="Max Sample Numbers" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item name="selection_num" label="Selection Number" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item name="num_samplers" label="Number of Samplers" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item name="num_evaluators" label="Number of Evaluators" rules={[{ required: true }]}> 
                        <Input type="number" />
                    </Form.Item>
                    <Form.Item style={{ textAlign: 'right' }}>
                        <Space>
                            <Button onClick={() => setRunModalVisible(false)}>Cancel</Button>
                            <Button type="primary" htmlType="submit">Start Optimization</Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
}

export default Dashboard;
