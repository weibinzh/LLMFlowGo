import React, { useState } from 'react';
import { Card, Button, Typography, message } from 'antd';
import { useNavigate } from 'react-router-dom';
import DAGVisualization from '../components/DAGVisualization';

const { Title, Text } = Typography;

const DAGSetup = () => {
  const navigate = useNavigate();
  const [nodeConfigs, setNodeConfigs] = useState({});
  const [dagParsed, setDagParsed] = useState(null);

  const handleConfigChange = (configs) => {
    setNodeConfigs(configs || {});
  };

  const handleDagChange = (parsed) => {
    setDagParsed(parsed || null);
  };

  const handleNext = () => {
    const hasWorkload = nodeConfigs && Object.keys(nodeConfigs).length > 0;
    const hasEdges = dagParsed && Array.isArray(dagParsed.edges) && dagParsed.edges.length > 0;
    if (!hasWorkload || !hasEdges) {
      message.warning('Please complete DAG workflow configuration (tasks and dependencies)');
      return;
    }

   // Combine into a unified dagConfig structure
    const dagConfig = {
      workload: nodeConfigs,
      edges: dagParsed.edges || [],
      task_dependencies: dagParsed.task_dependencies || {}
    };

    navigate('/environment', { state: { dagConfig } });
  };

  return (
    <div>
      <Title level={2}>DAG Setup</Title>
      <Card title="DAG Workflow Configuration" style={{ marginBottom: 16 }}>
        <DAGVisualization onConfigChange={handleConfigChange} onDagChange={handleDagChange} />
      </Card>
      <div style={{ textAlign: 'right' }}>
        <Button type="primary" onClick={handleNext}>Next: Server Environment Setup</Button>
      </div>
    </div>
  );
};

export default DAGSetup;
