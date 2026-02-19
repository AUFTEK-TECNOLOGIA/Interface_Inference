// src/pages/MachineLearning/ModelsPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardActionArea,
  CardContent,
  Typography,
  Button,
  Stack,
  Chip,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';


export default function ModelsPage() {
  const navigate = useNavigate();

  const cards = [
    {
      title: 'Meus Modelos',
      desc: 'Visualize e gerencie modelos treinados',
      icon: <FolderOpenIcon sx={{ fontSize: 60 }} color="primary" />,
      to: '/machine-learning/my-models',
      soon: false,
    },
    {
      title: 'Novo Modelo',
      desc: 'Treine um modelo a partir de sensores',
      icon: <ModelTrainingIcon sx={{ fontSize: 60 }} color="secondary" />,
      to: '/machine-learning/create-model', // já existe no seu app
      soon: false,
    },

  ];

  const handleCardClick = (card) => {
    if (card.soon) {
      // Evita navegação para rotas ainda não implementadas
      alert('Em breve');
      return;
    }
    navigate(card.to);
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
      {/* Header */}
      <Stack direction="row" spacing={2} alignItems="center" mb={3}>
        <Button
          startIcon={<ArrowBackIcon />}
          variant="outlined"
          size="small"
          onClick={() => navigate(-1)}
        >
          Voltar
        </Button>
        <Typography variant="h4" color="primary">
          Modelos
        </Typography>
      </Stack>

      {/* Grid de opções */}
      <Grid container spacing={4} justifyContent="center">
        {cards.map((card) => (
          <Grid key={card.title} item xs={12} sm={6} md={4}>
            <Card
              sx={{
                position: 'relative',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 6,
                },
              }}
            >
              {card.soon && (
                <Chip
                  label="Em breve"
                  color="default"
                  size="small"
                  sx={{ position: 'absolute', top: 12, right: 12 }}
                />
              )}
              <CardActionArea onClick={() => handleCardClick(card)}>
                <CardContent sx={{ textAlign: 'center', py: 6 }}>
                  {card.icon}
                  <Typography variant="h6" mt={2}>
                    {card.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" mt={1}>
                    {card.desc}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}